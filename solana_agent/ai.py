import asyncio
import datetime
import ntplib
import json
from typing import AsyncGenerator, List, Literal, Dict, Any, Callable
import uuid
import pandas as pd
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI
import inspect
import pytz
import requests
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_cloud.client import Zep as ZepCloud
from zep_python.client import Zep
from zep_python.client import AsyncZep
from zep_cloud.types import Message
from pinecone import Pinecone


class DocumentModel(BaseModel):
    id: str
    text: str


class MongoDatabase:
    def __init__(self, db_url: str, db_name: str):
        self._client = MongoClient(db_url)
        self.db = self._client[db_name]
        self.messages = self.db["messages"]
        self.kb = self.db["kb"]

    def save_message(self, user_id: str, metadata: Dict[str, Any]):
        metadata["user_id"] = user_id
        self.messages.insert_one(metadata)

    def clear_user_history(self, user_id: str):
        self.messages.delete_many({"user_id": user_id})

    def add_documents_to_kb(self, namespace: str, documents: List[DocumentModel]):
        for document in documents:
            storage = {}
            storage["namespace"] = namespace
            storage["reference"] = document.id
            storage["document"] = document.text
            storage["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            self.kb.insert_one(storage)

    def list_documents_in_kb(self, namespace: str) -> List[DocumentModel]:
        documents = self.kb.find({"namespace": namespace})
        return [
            DocumentModel(id=doc["reference"], text=doc["document"])
            for doc in documents
        ]


class AI:
    def __init__(
        self,
        openai_api_key: str,
        instructions: str,
        database: Any,
        zep_api_key: str = None,
        zep_base_url: str = None,
        perplexity_api_key: str = None,
        grok_api_key: str = None,
        pinecone_api_key: str = None,
        pinecone_index_name: str = None,
        pinecone_embed_model: Literal["llama-text-embed-v2"] = "llama-text-embed-v2",
        gemini_api_key: str = None,
        openai_base_url: str = None,
        tool_calling_model: str = "gpt-4o-mini",
        reasoning_model: str = "gpt-4o-mini",
    ):
        """Initialize a new AI assistant instance.

        Args:
            openai_api_key (str): OpenAI API key for core AI functionality
            instructions (str): Base behavioral instructions for the AI
            database (Any): Database instance for message/thread storage
            zep_api_key (str, optional): API key for Zep memory storage. Defaults to None
            zep_base_url (str, optional): Base URL for Zep API. Defaults to None
            perplexity_api_key (str, optional): API key for Perplexity search. Defaults to None
            grok_api_key (str, optional): API key for X/Twitter search via Grok. Defaults to None
            pinecone_api_key (str, optional): API key for Pinecone. Defaults to None
            pinecone_index_name (str, optional): Name of the Pinecone index. Defaults to None
            pinecone_embed_model (Literal["llama-text-embed-v2"], optional): Pinecone embedding model. Defaults to "llama-text-embed-v2"
            gemini_api_key (str, optional): API key for Gemini search. Defaults to None
            openai_base_url (str, optional): Base URL for OpenAI API. Defaults to None
            tool_calling_model (str, optional): Model for tool calling. Defaults to "gpt-4o-mini"
            reasoning_model (str, optional): Model for reasoning. Defaults to "gpt-4o-mini"
        Example:
            ```python
            ai = AI(
                openai_api_key="your-key",
                instructions="Be helpful and concise",
                database=MongoDatabase("mongodb://localhost", "ai_db")
            )
            ```
        Notes:
            - Requires valid OpenAI API key for core functionality
            - Supports any OpenAI compatible model for conversation
            - Requires valid Zep API key for memory storage
            - Database instance for storing messages and threads
            - Optional integrations for Perplexity, Pinecone, Gemini, and Grok
            - You must create the Pinecone index in the dashboard before using it
        """
        self._client = (
            OpenAI(api_key=openai_api_key, base_url=openai_base_url)
            if openai_base_url
            else OpenAI(api_key=openai_api_key)
        )
        self._memory_instructions = """
            You are a highly intelligent, context-aware conversational AI. When a user sends a query or statement, you should not only process the current input but also retrieve and integrate relevant context from their previous interactions. Use the memory data to:
            - Infer nuances in the user's intent.
            - Recall previous topics, preferences, or facts that might be relevant.
            - Provide a thoughtful, clear, and structured response.
            - Clarify ambiguous queries by relating them to known user history.

            Always be concise and ensure that your response maintains coherence across the conversation while respecting the user's context and previous data.
            You always take the Tool Result over the Memory Context in terms of priority.
        """
        self._instructions = instructions
        self._reasoning_instructions = self._memory_instructions + " " + instructions
        self._database: MongoDatabase = database
        self._accumulated_value_queue = asyncio.Queue()
        if zep_api_key and not zep_base_url:
            self._zep = AsyncZepCloud(api_key=zep_api_key)
            self._sync_zep = ZepCloud(api_key=zep_api_key)
        elif zep_api_key and zep_base_url:
            self._zep = AsyncZep(api_key=zep_api_key, base_url=zep_base_url)
            self._sync_zep = Zep(api_key=zep_api_key, base_url=zep_base_url)
        else:
            self._zep = None
            self._sync_zep = None
        self._perplexity_api_key = perplexity_api_key
        self._grok_api_key = grok_api_key
        self._gemini_api_key = gemini_api_key
        self._pinecone = (
            Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
        )
        self._pinecone_index_name = pinecone_index_name if pinecone_index_name else None
        self._pinecone_embedding_model = pinecone_embed_model
        self.kb = (
            self._pinecone.Index(
                self._pinecone_index_name) if self._pinecone else None
        )
        self._openai_base_url = openai_base_url
        self._tool_calling_model = tool_calling_model
        self._reasoning_model = reasoning_model
        self._tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Perform any cleanup actions here
        pass

    def csv_to_text(self, file, filename: str) -> str:
        """Convert a CSV file to a Markdown table text format optimized for LLM ingestion.

        Args:
            file (BinaryIO): The CSV file to convert to text.
            filename (str): The name of the CSV file.

        Returns:
            str: A Markdown formatted table representing the CSV data.

        Example:
            ```python
            result = ai.csv_to_text("data.csv")
            print(result)
            # Returns a Markdown table such as:
            # **Table: data**
            #
            # | Date       | Product  | Revenue |
            # | ---------- | -------- | ------- |
            # | 2024-01-01 | Widget A | $100    |
            # | 2024-01-02 | Widget B | $200    |
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
            The output format preserves the table structure allowing the LLM to understand column relationships and row data.
        """
        df = pd.read_csv(file)
        # Create header and separator rows for Markdown table
        header = "| " + " | ".join(df.columns.astype(str)) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        # Process each row in the dataframe
        rows = "\n".join("| " + " | ".join(map(str, row)) +
                         " |" for row in df.values)
        markdown_table = f"**Table: {filename}**\n\n{header}\n{separator}\n{rows}"
        return markdown_table

    # summarize tool - has to be sync
    def summarize(
        self,
        text: str,
        prompt: str = "Summarize the markdown table into a report, include important metrics and totals.",
        model: Literal["gemini-2.0-flash",
                       "gemini-1.5-pro"] = "gemini-1.5-pro",
    ) -> str:
        """Summarize text using Google's Gemini language model.

        Args:
            text (str): The text content to be summarized
            prompt (str, optional): The prompt to use for summarization. Defaults to "Summarize the markdown table into a report, include important metrics and totals."
            model (Literal["gemini-2.0-flash", "gemini-1.5-pro"], optional):
                Gemini model to use. Defaults to "gemini-1.5-pro"
                - gemini-2.0-flash: Faster, shorter summaries
                - gemini-1.5-pro: More detailed summaries

        Returns:
            str: Summarized text or error message if summarization fails

        Example:
            ```python
            summary = ai.summarize("Long article text here...")
            # Returns: "Concise summary of the article..."
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
            Requires valid Gemini API key to be configured.
        """
        try:
            client = OpenAI(
                api_key=self._gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": text},
                ],
            )

            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to summarize text. Error: {e}"

    def upload_csv_file_to_kb(
        self,
        file,
        filename: str,
        id: str = uuid.uuid4().hex,
        prompt: str = "Summarize the table into a report, include important metrics and totals.",
        namespace: str = "global",
        model: Literal["gemini-2.0-flash"] = "gemini-2.0-flash",
    ):
        """Upload and process a CSV file into the knowledge base with AI summarization.

        Args:
            file (BinaryIO): The CSV file to upload and process
            filename (str): The name of the CSV file
            id (str, optional): Unique identifier for the document. Defaults to a random UUID.
            prompt (str, optional): Custom prompt for summarization. Defaults to "Summarize the table into a report, include important metrics and totals."
            namespace (str, optional): Knowledge base namespace. Defaults to "global".
            model (Literal["gemini-2.0-flash"], optional): Gemini model for summarization. Defaults to "gemini-2.0-flash".

        Example:
            ```python
            ai.upload_csv_file(
                file=open("data.csv", "rb"),
                filename="data.csv",
            )
            ```

        Note:
            - Converts CSV to Markdown table format
            - Uses Gemini AI to generate a summary - total of 1M context tokens
            - Stores summary in Pinecone knowledge base
            - Requires configured Pinecone index
            - Supports custom prompts for targeted summaries
        """
        csv_text = self.csv_to_text(file, filename)
        document = self.summarize(csv_text, prompt, model)
        self.add_documents_to_kb(
            documents=[DocumentModel(id=id, text=document)], namespace=namespace
        )

    def search_kb(
        self,
        query: str,
        namespace: str = "global",
        rerank_model: Literal["cohere-rerank-3.5"] = "cohere-rerank-3.5",
        inner_limit: int = 10,
        limit: int = 3,
    ) -> str:
        """Search Pinecone knowledge base.

        Args:
            query (str): Search query to find relevant documents
            namespace (str, optional): Namespace of the Pinecone to search. Defaults to "global".
            rerank_model (Literal["cohere-rerank-3.5"], optional): Rerank model on Pinecone. Defaults to "cohere-rerank-3.5".
            inner_limit (int, optional): Maximum number of results to rerank. Defaults to 10.
            limit (int, optional): Maximum number of results to return. Defaults to 3.

        Returns:
            str: JSON string of matched documents or error message

        Example:
            ```python
            results = ai.search_kb("machine learning basics", "user123")
            # Returns: '["Document 1", "Document 2", ...]'
            ```

        Note:
            - Requires configured Pinecone index
            - Returns error message string if search fails
        """
        try:
            embedding = self._pinecone.inference.embed(
                model=self._pinecone_embedding_model,
                inputs=[query],
                parameters={"input_type": "query"},
            )
            search_results = self.kb.query(
                vector=embedding[0].values,
                top_k=inner_limit,
                include_metadata=False,
                include_values=False,
                namespace=namespace,
            )
            matches = search_results.matches
            ids = []
            for match in matches:
                ids.append(match.id)
            docs = []
            for id in ids:
                document = self._database.kb.find_one({"reference": id})
                docs.append(document["document"])
            try:
                reranked_docs = self._pinecone.inference.rerank(
                    model=rerank_model,
                    query=query,
                    documents=docs,
                    top_n=limit,
                )
                new_docs = []
                for doc in reranked_docs.data:
                    new_docs.append(docs[doc.index])
                return json.dumps(new_docs)
            except Exception:
                return json.dumps(docs[:limit])
        except Exception as e:
            return f"Failed to search KB. Error: {e}"

    def list_documents_in_kb(self, namespace: str = "global") -> List[DocumentModel]:
        """List all documents stored in the Pinecone knowledge base.

        Args:
            namespace (str, optional): Namespace of the Pinecone index to search. Defaults to "global".

        Returns:
            List[DocumentModel]: List of documents stored in the knowledge base

        Example:
            ```python
            documents = ai.list_documents_in_kb("user123")
            for doc in documents:
                print(doc)
            # Returns: "Document 1", "Document 2", ...
            ```

        Note:
            - Requires Pinecone index to be configured
        """
        return self._database.list_documents_in_kb(namespace)

    def add_documents_to_kb(
        self,
        documents: List[DocumentModel],
        namespace: str = "global",
    ):
        """Add documents to the Pinecone knowledge base.

        Args:
            documents (List[DocumentModel]): List of documents to add to the knowledge base
            namespace (str): Namespace of the Pinecone index to search. Defaults to "global".

        Example:
            ```python
            docs = [
                {"id": "doc1", "text": "Document 1"},
                {"id": "doc2", "text": "Document 2"},
            ]
            ai.add_documents_to_kb(docs, "user123")
            ```

        Note:
            - Requires Pinecone index to be configured
        """
        embeddings = self._pinecone.inference.embed(
            model=self._pinecone_embedding_model,
            inputs=[d.text for d in documents],
            parameters={"input_type": "passage", "truncate": "END"},
        )

        vectors = []
        for d, e in zip(documents, embeddings):
            vectors.append(
                {
                    "id": d.id,
                    "values": e["values"],
                }
            )

        self.kb.upsert(
            vectors=vectors,
            namespace=namespace,
        )

        self._database.add_documents_to_kb(namespace, documents)

    def delete_document_from_kb(self, id: str, user_id: str = "global"):
        """Delete a document from the Pinecone knowledge base.

        Args:
            id (str): Unique identifier for the document
            user_id (str): Unique identifier for the user. Defaults to "global".

        Example:
            ```python
            ai.delete_document_from_kb("user123", "document_id")
            ```

        Note:
            - Requires Pinecone index to be configured
        """
        self.kb.delete(ids=[id], namespace=user_id)
        self._database.kb.delete_one({"reference": id})

    def check_time(self, timezone: str) -> str:
        """Get current UTC time formatted as a string via Cloudflare's NTP service.

        Args:
            timezone (str): Timezone to convert the time to (e.g., "America/New_York")

        Returns:
            str: Current time in the requested timezone in format 'YYYY-MM-DD HH:MM:SS'

        Example:
            ```python
            time = ai.check_time("America/New_York")
            # Returns: "The current time in America/New_York is 2025-02-26 10:30:45"
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
            Fetches time over NTP from Cloudflare's time server (time.cloudflare.com).
        """
        try:
            # Request time from Cloudflare's NTP server
            client = ntplib.NTPClient()
            response = client.request("time.cloudflare.com", version=3)

            # Get UTC time from NTP response
            utc_dt = datetime.datetime.fromtimestamp(
                response.tx_time, datetime.timezone.utc
            )

            # Convert to requested timezone
            try:
                tz = pytz.timezone(timezone)
                local_dt = utc_dt.astimezone(tz)
                formatted_time = local_dt.strftime("%Y-%m-%d %H:%M:%S")
                return f"The current time in {timezone} is {formatted_time}"
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone}'. Please use a valid timezone like 'America/New_York'."

        except Exception as e:
            return f"Error getting the current time: {e}"

    # has to be sync for tool
    def get_memory_context(
        self,
        user_id: str,
    ) -> str:
        """Retrieve contextual memory for a specific user from Zep memory storage.

        Args:
            user_id (str): Unique identifier for the user whose memory context to retrieve

        Returns:
            str: User's memory context or error message if retrieval fails

        Example:
            ```python
            context = ai.get_memory_context("user123")
            print(context)
            # Returns: "User previously mentioned having 3 dogs and living in London"
            ```

        Note:
            - This is a synchronous tool method required for OpenAI function calling
            - Requires Zep integration to be configured with valid API key
            - Returns error message if Zep is not configured or retrieval fails
            - Useful for maintaining conversation context across sessions
        """
        try:
            memory = self._sync_zep.memory.get(session_id=user_id)
            return memory.context
        except Exception:
            return ""

    # search internet tool - has to be sync
    def search_internet(
        self,
        query: str,
        model: Literal[
            "sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning"
        ] = "sonar",
    ) -> str:
        """Search the internet using Perplexity AI API.

        Args:
            query (str): Search query string
            model (Literal["sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning"], optional):
                Perplexity model to use. Defaults to "sonar"
                - sonar: Fast, general-purpose search
                - sonar-pro: Enhanced search capabilities
                - sonar-reasoning-pro: Advanced reasoning with search
                - sonar-reasoning: Basic reasoning with search

        Returns:
            str: Search results or error message if search fails

        Example:
            ```python
            result = ai.search_internet(
                query="Latest AI developments",
                model="sonar-reasoning-pro"
            )
            # Returns: "Detailed search results about AI..."
            ```

        Note:
            Requires valid Perplexity API key to be configured.
            This is a synchronous tool method required for OpenAI function calling.
        """
        try:
            url = "https://api.perplexity.ai/chat/completions"

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer the user's query.",
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            }
            headers = {
                "Authorization": f"Bearer {self._perplexity_api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                return (
                    f"Failed to search Perplexity. Status code: {response.status_code}"
                )
        except Exception as e:
            return f"Failed to search Perplexity. Error: {e}"

    # x search tool - has to be sync
    def search_x(
        self, query: str, model: Literal["grok-2-latest"] = "grok-2-latest"
    ) -> str:
        try:
            """Search X (formerly Twitter) using Grok API integration.

            Args:
                query (str): Search query to find relevant X posts
                model (Literal["grok-2-latest"], optional): Grok model to use. Defaults to "grok-2-latest"

            Returns:
                str: Search results from X or error message if search fails

            Example:
                ```python
                result = ai.search_x("AI announcements")
                # Returns: "Recent relevant X posts about AI announcements..."
                ```

            Note:
                This is a synchronous tool method required for OpenAI function calling.
                Requires valid Grok API key to be configured.
                Returns error message string if API call fails.
            """
            client = OpenAI(api_key=self._grok_api_key,
                            base_url="https://api.x.ai/v1")

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer the user's query.",
                    },
                    {"role": "user", "content": query},
                ],
            )

            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to search X. Error: {e}"

    async def clear_user_history(self, user_id: str):
        """Clear stored conversation history for a specific user.

        Args:
            user_id (str): Unique identifier for the user whose history should be cleared

        Example:
            ```python
            await ai.clear_user_history("user123")
            # Clears all stored messages, facts, and threads for user123
            ```

        Note:
            This is an async method and must be awaited.
        """
        try:
            self._database.clear_user_history(user_id)
        except Exception:
            pass
        try:
            await self.delete_memory(user_id)
        except Exception:
            pass

    async def delete_memory(self, user_id: str):
        """Delete memory for a specific user from Zep memory.

        Args:
            user_id (str): Unique identifier for the user whose facts should be deleted

        Example:
            ```python
            await ai.delete_facts("user123")
            # Deletes all stored facts for user123
            ```

        Note:
            This is an async method and must be awaited.
            Requires Zep integration to be configured.
            No-op if Zep is not configured.
        """
        if self._zep:
            await self._zep.memory.delete(session_id=user_id)
            await self._zep.user.delete(user_id=user_id)

    async def _listen(self, audio_content: bytes, input_format: str) -> str:
        transcription = self._client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"file.{input_format}", audio_content),
        )
        return transcription.text

    async def text(self, user_id: str, user_text: str) -> AsyncGenerator[str, None]:
        """Process text input and stream AI responses asynchronously.

        Args:
            user_id (str): Unique identifier for the user/conversation.
            user_text (str): Text input from user to process.

        Returns:
            AsyncGenerator[str, None]: Stream of response text chunks (including tool call results).

        Example:
            ```python
            async for chunk in ai.text("user123", "What is machine learning?"):
                print(chunk, end="")  # Prints response as it streams
            ```

        Note:
            - Maintains conversation thread using OpenAI's thread system.
            - Stores messages in configured database (MongoDB/SQLite).
            - Integrates with Zep memory if configured.
            - Supports tool calls by aggregating and executing them as their arguments stream in.
        """
        self._accumulated_value_queue = asyncio.Queue()
        final_tool_calls = {}  # Accumulate tool call deltas
        final_response = ""

        if self._zep:
            messages = [
                Message(
                    role="user",
                    role_type="user",
                    content=user_text,
                ),
            ]
            await self._zep.memory.add(session_id=user_id, messages=messages)

        async def stream_processor():
            memory = self.get_memory_context(user_id)
            regular_content = ""
            response = self._client.chat.completions.create(
                model=self._tool_calling_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._instructions,
                    },
                    {
                        "role": "user",
                        "content": user_text,
                    },
                ],
                tools=self._tools,
                stream=True,
            )
            for chunk in response:
                result = ""
                delta = chunk.choices[0].delta

                # Process tool call deltas (if any)
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        index = tool_call.index
                        if tool_call.function.name:
                            # Initialize a new tool call record
                            final_tool_calls[index] = {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or "",
                            }
                        elif tool_call.function.arguments:
                            # Append additional arguments if provided in subsequent chunks
                            final_tool_calls[index]["arguments"] += (
                                tool_call.function.arguments
                            )

                            try:
                                args = json.loads(
                                    final_tool_calls[index]["arguments"])
                                func = getattr(
                                    self, final_tool_calls[index]["name"])
                                # Execute the tool call (synchronously; adjust if async is needed)
                                result = func(**args)
                                response = self._client.chat.completions.create(
                                    model=self._reasoning_model,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"Rules: {self._reasoning_instructions}, Tool Result: {result}, Memory Context: {memory}",
                                        },
                                        {
                                            "role": "user",
                                            "content": user_text,
                                        },
                                    ],
                                    stream=True,
                                )
                                for chunk in response:
                                    delta = chunk.choices[0].delta

                                    if delta.content is not None:
                                        await self._accumulated_value_queue.put(
                                            delta.content
                                        )
                                # Remove the cached tool call info so it doesn't block future calls
                                del final_tool_calls[index]
                            except json.JSONDecodeError:
                                # If the accumulated arguments aren't valid yet, wait for more chunks.
                                continue

                # Process regular response content
                if delta.content is not None:
                    regular_content += (
                        delta.content
                    )  # Accumulate instead of directly sending

            # After processing all chunks from the first response
            if regular_content:  # Only if we have regular content
                # Format the regular content with memory context, similar to tool results
                response = self._client.chat.completions.create(
                    model=self._reasoning_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Rules: {self._reasoning_instructions}, Memory Context: {memory}",
                        },
                        {
                            "role": "user",
                            "content": user_text,
                        },
                    ],
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        await self._accumulated_value_queue.put(delta.content)

        # Start the stream processor as a background task
        asyncio.create_task(stream_processor())

        # Yield values from the queue as they become available.
        while True:
            try:
                value = await asyncio.wait_for(
                    self._accumulated_value_queue.get(), timeout=0.1
                )
                if value is not None:
                    final_response += value
                    yield value
            except asyncio.TimeoutError:
                # Break only if the queue is empty (assuming stream ended)
                if self._accumulated_value_queue.empty():
                    break

        # Save the conversation to the database and Zep memory (if configured)
        metadata = {
            "user_id": user_id,
            "message": user_text,
            "response": final_response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        self._database.save_message(user_id, metadata)

        if self._zep:
            messages = [
                Message(
                    role="assistant",
                    role_type="assistant",
                    content=final_response,
                ),
            ]
            await self._zep.memory.add(session_id=user_id, messages=messages)

    async def conversation(
        self,
        user_id: str,
        audio_bytes: bytes,
        voice: Literal["alloy", "echo", "fable",
                       "onyx", "nova", "shimmer"] = "nova",
        input_format: Literal[
            "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"
        ] = "mp4",
        response_format: Literal["mp3", "opus",
                                 "aac", "flac", "wav", "pcm"] = "aac",
    ) -> AsyncGenerator[bytes, None]:
        """Process voice conversations and stream AI audio responses asynchronously.

        Args:
            user_id (str): Unique identifier for the user/conversation
            audio_bytes (bytes): Raw audio input bytes to process
            voice (Literal, optional): OpenAI TTS voice to use. Defaults to "nova"
            input_format (Literal, optional): Input audio format. Defaults to "mp4"
            response_format (Literal, optional): Output audio format. Defaults to "aac"

        Returns:
            AsyncGenerator[bytes, None]: Stream of audio response chunks

        Example:
            ```python
            async with open('input.mp4', 'rb') as f:
                audio_data = f.read()
                async for chunk in ai.conversation(
                    "user123",
                    audio_data,
                    voice="nova",
                    input_format="mp4",
                    response_format="aac"
                ):
                    # Process or save audio chunks
                    await process_audio_chunk(chunk)
            ```

        Note:
            - Converts audio to text using Whisper
            - Maintains conversation thread using OpenAI
            - Stores conversation in database
            - Integrates with Zep memory if configured
            - Streams audio response using OpenAI TTS
        """

        transcript = await self._listen(audio_bytes, input_format)
        self._accumulated_value_queue = asyncio.Queue()
        final_tool_calls = {}  # Accumulate tool call deltas
        final_response = ""

        if self._zep:
            messages = [
                Message(
                    role="user",
                    role_type="user",
                    content=transcript,
                ),
            ]
            await self._zep.memory.add(session_id=user_id, messages=messages)

        async def stream_processor():
            memory = self.get_memory_context(user_id)
            regular_content = ""  # Add this to accumulate regular content
            response = self._client.chat.completions.create(
                model=self._tool_calling_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._instructions,
                    },
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ],
                tools=self._tools,
                stream=True,
            )
            for chunk in response:
                result = ""
                delta = chunk.choices[0].delta

                # Process tool call deltas (if any)
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        index = tool_call.index
                        if tool_call.function.name:
                            # Initialize a new tool call record
                            final_tool_calls[index] = {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or "",
                            }
                        elif tool_call.function.arguments:
                            # Append additional arguments if provided in subsequent chunks
                            final_tool_calls[index]["arguments"] += (
                                tool_call.function.arguments
                            )

                            try:
                                args = json.loads(
                                    final_tool_calls[index]["arguments"])
                                func = getattr(
                                    self, final_tool_calls[index]["name"])
                                # Execute the tool call (synchronously; adjust if async is needed)
                                result = func(**args)
                                response = self._client.chat.completions.create(
                                    model=self._reasoning_model,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"Rules: {self._reasoning_instructions}, Tool Result: {result}, Memory Context: {memory}",
                                        },
                                        {
                                            "role": "user",
                                            "content": transcript,
                                        },
                                    ],
                                    stream=True,
                                )
                                for chunk in response:
                                    delta = chunk.choices[0].delta

                                    if delta.content is not None:
                                        await self._accumulated_value_queue.put(
                                            delta.content
                                        )
                                # Remove the cached tool call info so it doesn't block future calls
                                del final_tool_calls[index]
                            except json.JSONDecodeError:
                                # If the accumulated arguments aren't valid yet, wait for more chunks.
                                continue

                # Process regular response content
                if delta.content is not None:
                    regular_content += (
                        delta.content
                    )  # Accumulate instead of directly sending

            # After processing all chunks from the first response
            if regular_content:  # Only if we have regular content
                # Format the regular content with memory context, similar to tool results
                response = self._client.chat.completions.create(
                    model=self._reasoning_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Rules: {self._reasoning_instructions}, Memory Context: {memory}",
                        },
                        {
                            "role": "user",
                            "content": transcript,
                        },
                    ],
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        await self._accumulated_value_queue.put(delta.content)

        # Start the stream processor as a background task
        asyncio.create_task(stream_processor())

        # Yield values from the queue as they become available.
        while True:
            try:
                value = await asyncio.wait_for(
                    self._accumulated_value_queue.get(), timeout=0.1
                )
                if value is not None:
                    final_response += value
                    yield value
            except asyncio.TimeoutError:
                # Break only if the queue is empty (assuming stream ended)
                if self._accumulated_value_queue.empty():
                    break

        # Save the conversation to the database and Zep memory (if configured)
        metadata = {
            "user_id": user_id,
            "message": transcript,
            "response": final_response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        self._database.save_message(user_id, metadata)

        if self._zep:
            messages = [
                Message(
                    role="assistant",
                    role_type="assistant",
                    content=final_response,
                ),
            ]
            await self._zep.memory.add(session_id=user_id, messages=messages)

        # Generate and stream the audio response
        with self._client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=final_response,
            response_format=response_format,
        ) as response:
            for chunk in response.iter_bytes(1024):
                yield chunk

    def add_tool(self, func: Callable):
        """Register a custom function as an AI tool using decorator pattern.

        Args:
            func (Callable): Function to register as a tool. Must have docstring and type hints.

        Returns:
            Callable: The decorated function

        Example:
            ```python
            @ai.add_tool
            def custom_search(query: str) -> str:
                '''Search custom data source.

                Args:
                    query (str): Search query

                Returns:
                    str: Search results
                '''
                return "Custom search results"
            ```

        Note:
            - Function must have proper docstring for tool description
            - Parameters should have type hints
            - Tool becomes available to AI for function calling
            - Parameters are automatically converted to JSON schema
        """
        sig = inspect.signature(func)
        parameters = {"type": "object", "properties": {}, "required": []}
        for name, param in sig.parameters.items():
            parameters["properties"][name] = {
                "type": "string", "description": "foo"}
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)
        tool_config = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": parameters,
            },
        }
        self._tools.append(tool_config)
        # Attach the function to the instance so getattr can find it later.
        setattr(self, func.__name__, func)
        return func


class MultiAgentSystem:
    """A multi-agent system that coordinates specialized AI agents with handoff capabilities."""

    def __init__(self, database: MongoDatabase, router_model: str = "gpt-4o"):
        """Initialize the multi-agent system with a shared database.

        Args:
            database (MongoDatabase): Shared MongoDB database instance
            router_model (str, optional): Model to use for routing decisions. Defaults to "gpt-4o".
        """
        self.agents = {}  # name -> AI instance
        self.specializations = {}  # name -> description
        self.database = database
        self.router_model = router_model

        # Ensure handoffs collection exists
        if "handoffs" not in self.database.db.list_collection_names():
            self.database.db.create_collection("handoffs")
        self.handoffs = self.database.db["handoffs"]

        print(
            f"MultiAgentSystem initialized with router model: {router_model}")

    def register(self, name: str, agent: AI, specialization: str):
        """Register a specialized agent with the multi-agent system."""
        # Add the agent to the system first
        self.agents[name] = agent
        self.specializations[name] = specialization

        print(
            f"Registered agent: {name}, specialization: {specialization[:50]}...")
        print(f"Current agents: {list(self.agents.keys())}")

        # We need to refresh handoff tools for ALL agents whenever a new one is registered
        self._update_all_agent_tools()

    def _update_all_agent_tools(self):
        """Update all agents with current handoff capabilities."""
        # For each registered agent, update its handoff tool
        for agent_name, agent in self.agents.items():
            # Get other agents that this agent can hand off to
            available_targets = [
                name for name in self.agents.keys() if name != agent_name
            ]
            specialization = self.specializations[agent_name]

            # First remove any existing handoff tool if present
            agent._tools = [
                t for t in agent._tools if t["function"]["name"] != "request_handoff"
            ]

            # Fix: Create a function factory with proper closure to capture current agent name
            def create_handoff_tool(current_agent_name, available_targets_list):
                def request_handoff(target_agent: str, reason: str) -> str:
                    """Request an immediate handoff to another specialized agent.
                    This is an INTERNAL SYSTEM TOOL. The user will NOT see your reasoning about the handoff.
                    Use this tool IMMEDIATELY when a query is outside your expertise.

                    Args:
                        target_agent (str): Name of agent to transfer to (choices: {', '.join(available_targets_list)})
                        reason (str): Brief explanation of why this question requires the specialist

                    Returns:
                        str: Internal handoff marker (not shown to user)
                    """
                    # Prevent self-handoffs
                    if target_agent == current_agent_name:
                        print(
                            f"[HANDOFF ERROR] Agent {current_agent_name} attempted to hand off to itself"
                        )
                        if available_targets_list:
                            target_agent = available_targets_list[0]
                            print(
                                f"[HANDOFF CORRECTION] Redirecting to {target_agent} instead"
                            )
                        else:
                            print(
                                "[HANDOFF ERROR] No other agents available to hand off to"
                            )
                            return ""

                    # Validate target agent exists
                    if target_agent not in self.agents:
                        print(
                            f"[HANDOFF WARNING] Invalid target '{target_agent}'")
                        if available_targets_list:
                            original_target = target_agent
                            target_agent = available_targets_list[0]
                            print(
                                f"[HANDOFF CORRECTION] Redirecting from '{original_target}' to '{target_agent}'"
                            )
                        else:
                            print(
                                "[HANDOFF ERROR] No valid target agents available")
                            return ""

                    print(
                        f"[HANDOFF TOOL CALLED] {current_agent_name} -> {target_agent}: {reason}"
                    )
                    # Return ONLY the marker - no extra explanations
                    return f"__HANDOFF__{target_agent}__{reason}__"

                return request_handoff

            # Use the factory to create a properly-bound tool function
            handoff_tool = create_handoff_tool(agent_name, available_targets)

            # Now add the updated handoff tool with proper closure
            agent.add_tool(handoff_tool)

            # Add critical handoff instructions to agent
            handoff_instructions = f"""
            You are specialized in {specialization}.
            
            STRICT HANDOFF RULES:
            1. For compound questions that involve topics outside your expertise:
            * IMMEDIATELY use the request_handoff tool with NO preliminary explanation
            * DO NOT attempt to answer ANY part of the question before handoff
            * Let the specialist handle the ENTIRE question
            
            2. Valid handoff targets: {', '.join(available_targets)}
            """

            # Add specialized guidance based on available agents
            if available_targets:
                handoff_guidance = "\n   Handoff guidance:\n"
                for target in available_targets:
                    target_specialization = self.specializations.get(
                        target, "")
                    # Add first 50 chars of specialization as brief description
                    short_desc = target_specialization[:50] + (
                        "..." if len(target_specialization) > 50 else ""
                    )
                    handoff_guidance += (
                        f'   * For questions about {short_desc}, use "{target}"\n'
                    )
                handoff_instructions += handoff_guidance

            print(
                f"Updated handoff capabilities for {agent_name} with targets: {available_targets}"
            )

    async def process(self, user_id: str, user_text: str) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle handoffs."""
        try:
            # Check if any agents are registered
            if not self.agents:
                yield "Error: No agents are registered with the system. Please register at least one agent first."
                return

            # Get routing decision
            first_agent = next(iter(self.agents.values()))
            agent_name = await self._get_routing_decision(first_agent, user_text)
            current_agent = self.agents[agent_name]
            print(f"Starting conversation with agent: {agent_name}")

            # For handoff detection - use minimal sliding window
            handoff_detector = "__HANDOFF__"
            sliding_window = ""
            max_window_size = len(handoff_detector) + 50
            found_handoff = False

            # Process initial agent's response with minimal buffering
            async for chunk in current_agent.text(user_id, user_text):
                if found_handoff:
                    continue  # Skip remaining chunks after handoff

                # CRITICAL: Yield each chunk immediately with no preprocessing
                yield chunk
                # Force context switch to allow client to receive chunk
                await asyncio.sleep(0)

                # Simple handoff detection with minimal processing
                sliding_window += chunk
                if len(sliding_window) > max_window_size:
                    sliding_window = sliding_window[-max_window_size:]

                if handoff_detector in sliding_window and not found_handoff:
                    found_handoff = True
                    print("[HANDOFF DETECTED]")

                    # Process handoff with minimal delay
                    parts = sliding_window.split(handoff_detector, 1)
                    handoff_parts = parts[1].split(
                        "__", 2) if len(parts) > 1 else []

                    if len(handoff_parts) >= 2:
                        target_name = handoff_parts[0]
                        reason = handoff_parts[1]

                        # Record handoff without waiting
                        asyncio.create_task(
                            self._record_handoff(
                                user_id, agent_name, target_name, reason, user_text
                            )
                        )

                        # Process with target agent
                        print(f"[HANDOFF] Forwarding to {target_name}")
                        handoff_query = (
                            user_text  # Simplified query to reduce processing
                        )

                        # Stream directly from target agent
                        async for new_chunk in self.agents[target_name].text(
                            user_id, handoff_query
                        ):
                            yield new_chunk
                            # Force immediate delivery of each chunk
                            await asyncio.sleep(0)
                        return

        except Exception as e:
            print(f"Error in multi-agent processing: {str(e)}")
            import traceback

            print(traceback.format_exc())
            yield "\n\nI apologize for the technical difficulty.\n\n"

    async def _get_routing_decision(self, agent, user_text):
        """Get routing decision in parallel to reduce latency."""
        enhanced_prompt = f"""
        Analyze this user query carefully to determine the MOST APPROPRIATE specialist.
        
        User query: "{user_text}"
        
        Available specialists:
        {json.dumps(self.specializations, indent=2)}
        
        CRITICAL ROUTING INSTRUCTIONS:
        1. For compound questions with multiple aspects spanning different domains,
           choose the specialist who should address the CONCEPTUAL or EDUCATIONAL aspects first.
        
        2. Choose implementation specialists (technical, development, coding) only when
           the query is PURELY about implementation with no conceptual explanation needed.
        
        3. When a query involves a SEQUENCE (like "explain X and then do Y"),
           prioritize the specialist handling the FIRST part of the sequence.
        
        Return ONLY the name of the single most appropriate specialist.
        """

        # Route to appropriate agent
        router_response = agent._client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.2,
        )

        # Extract the selected agent
        raw_response = router_response.choices[0].message.content.strip()
        print(f"Router model raw response: '{raw_response}'")

        return self._match_agent_name(raw_response)

    async def _record_handoff(self, user_id, from_agent, to_agent, reason, query):
        """Record handoff in database without blocking the main flow."""
        self.handoffs.insert_one(
            {
                "user_id": user_id,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "query": query,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
            }
        )

    def _match_agent_name(self, raw_response):
        """Match router response to an actual agent name."""
        # Exact match (priority)
        if raw_response in self.agents:
            return raw_response

        # Case-insensitive match
        for name in self.agents:
            if name.lower() == raw_response.lower():
                return name

        # Partial match
        for name in self.agents:
            if name.lower() in raw_response.lower():
                return name

        # Fallback to first agent
        return list(self.agents.keys())[0]
