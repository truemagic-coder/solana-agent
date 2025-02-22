import asyncio
import datetime
import json
from typing import AsyncGenerator, Literal, Optional, Dict, Any, Callable
import uuid
import cohere
import pandas as pd
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI
from openai import AssistantEventHandler
from openai.types.beta.threads import TextDelta, Text
from typing_extensions import override
import inspect
import requests
from zep_cloud.client import AsyncZep
from zep_cloud.client import Zep
from zep_cloud.types import Message
from pinecone import Pinecone


class EventHandler(AssistantEventHandler):
    def __init__(self, tool_handlers, ai_instance):
        super().__init__()
        self._tool_handlers = tool_handlers
        self._ai_instance = ai_instance

    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text):
        asyncio.create_task(
            self._ai_instance._accumulated_value_queue.put(delta.value))

    @override
    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self._ai_instance._handle_requires_action(event.data, run_id)


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class MongoDatabase:
    def __init__(self, db_url: str, db_name: str):
        self._client = MongoClient(db_url)
        self.db = self._client[db_name]
        self._threads = self.db["threads"]
        self.messages = self.db["messages"]
        self.kb = self.db["kb"]
        self.vector_stores = self.db["vector_stores"]
        self.files = self.db["files"]

    def save_thread_id(self, user_id: str, thread_id: str):
        self._threads.insert_one({"thread_id": thread_id, "user_id": user_id})

    def get_thread_id(self, user_id: str) -> Optional[str]:
        document = self._threads.find_one({"user_id": user_id})
        return document["thread_id"] if document else None

    def save_message(self, user_id: str, metadata: Dict[str, Any]):
        metadata["user_id"] = user_id
        self.messages.insert_one(metadata)

    def delete_all_threads(self):
        self._threads.delete_many({})

    def clear_user_history(self, user_id: str):
        self.messages.delete_many({"user_id": user_id})
        self._threads.delete_one({"user_id": user_id})

    def add_document_to_kb(self, id: str, namespace: str, document: str):
        storage = {}
        storage["namespace"] = namespace
        storage["reference"] = id
        storage["document"] = document
        storage["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
        self.kb.insert_one(storage)

    def get_vector_store_id(self) -> str | None:
        document = self.vector_stores.find_one()
        return document["vector_store_id"] if document else None

    def save_vector_store_id(self, vector_store_id: str):
        self.vector_stores.insert_one({"vector_store_id": vector_store_id})

    def delete_vector_store_id(self, vector_store_id: str):
        self.vector_stores.delete_one({"vector_store_id": vector_store_id})

    def add_file(self, file_id: str):
        self.files.insert_one({"file_id": file_id})

    def delete_file(self, file_id: str):
        self.files.delete_one({"file_id": file_id})


class AI:
    def __init__(
        self,
        openai_api_key: str,
        name: str,
        instructions: str,
        database: Any,
        zep_api_key: str = None,
        perplexity_api_key: str = None,
        grok_api_key: str = None,
        pinecone_api_key: str = None,
        pinecone_index_name: str = None,
        cohere_api_key: str = None,
        cohere_model: Literal["rerank-v3.5"] = "rerank-v3.5",
        gemini_api_key: str = None,
        code_interpreter: bool = False,
        file_search: bool = False,
        openai_assistant_model: Literal["gpt-4o-mini",
                                        "gpt-4o"] = "gpt-4o-mini",
        openai_embedding_model: Literal[
            "text-embedding-3-small", "text-embedding-3-large"
        ] = "text-embedding-3-large",
    ):
        """Initialize a new AI assistant with memory and tool integration capabilities.

        Args:
            openai_api_key (str): OpenAI API key for core AI functionality
            name (str): Name identifier for the assistant
            instructions (str): Base behavioral instructions for the AI
            database (Any): Database instance for message/thread storage
            zep_api_key (str, optional): API key for Zep memory integration. Defaults to None
            perplexity_api_key (str, optional): API key for Perplexity search. Defaults to None
            grok_api_key (str, optional): API key for X/Twitter search via Grok. Defaults to None
            pinecone_api_key (str, optional): API key for Pinecone. Defaults to None
            pinecone_index_name (str, optional): Name of the Pinecone index. Defaults to None
            cohere_api_key (str, optional): API key for Cohere search. Defaults to None
            cohere_model (Literal["rerank-v3.5"], optional): Cohere model for reranking. Defaults to "rerank-v3.5"
            gemini_api_key (str, optional): API key for Gemini search. Defaults to None
            code_interpreter (bool, optional): Enable code interpretation. Defaults to False
            file_search (bool, optional): Enable file search tool. Defaults to False
            openai_assistant_model (Literal["gpt-4o-mini", "gpt-4o"], optional): OpenAI model for assistant. Defaults to "gpt-4o-mini"
            openai_embedding_model (Literal["text-embedding-3-small", "text-embedding-3-large"], optional): OpenAI model for text embedding. Defaults to "text-embedding-3-large"

        Example:
            ```python
            ai = AI(
                openai_api_key="your-key",
                name="Assistant",
                instructions="Be helpful and concise",
                database=MongoDatabase("mongodb://localhost", "ai_db"),
            )
            ```
        Notes:
            - Requires valid OpenAI API key for core functionality
            - Database instance for storing messages and threads
            - Optional integrations for Zep, Perplexity, Pinecone, Cohere, Gemini, and Grok
            - Supports code interpretation and custom tool functions
            - You must create the Pinecone index in the dashboard before using it
        """
        self._client = OpenAI(api_key=openai_api_key)
        self._name = name
        self._instructions = instructions
        self._openai_assistant_model = openai_assistant_model
        self._openai_embedding_model = openai_embedding_model
        self._file_search = file_search
        if file_search:
            self._tools = (
                [
                    {"type": "code_interpreter"},
                    {"type": "file_search"},
                ]
                if code_interpreter
                else [{"type": "file_search"}]
            )
        else:
            self._tools = [{"type": "code_interpreter"}
                           ] if code_interpreter else []

        self._tool_handlers = {}
        self._assistant_id = None
        self._database: MongoDatabase = database
        self._accumulated_value_queue = asyncio.Queue()
        self._zep = AsyncZep(api_key=zep_api_key) if zep_api_key else None
        self._sync_zep = Zep(api_key=zep_api_key) if zep_api_key else None
        self._perplexity_api_key = perplexity_api_key
        self._grok_api_key = grok_api_key
        self._gemini_api_key = gemini_api_key
        self._pinecone = (
            Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
        )
        self._pinecone_index_name = pinecone_index_name if pinecone_index_name else None
        self.kb = (
            self._pinecone.Index(
                self._pinecone_index_name) if self._pinecone else None
        )
        self._co = cohere.ClientV2(
            api_key=cohere_api_key) if cohere_api_key else None
        self._co_model = cohere_model if cohere_api_key else None

    async def __aenter__(self):
        assistants = self._client.beta.assistants.list()
        existing_assistant = next(
            (a for a in assistants if a.name == self._name), None)

        if existing_assistant:
            self._assistant_id = existing_assistant.id
        else:
            self._assistant_id = self._client.beta.assistants.create(
                name=self._name,
                instructions=self._instructions,
                tools=self._tools,
                model=self._openai_assistant_model,
            ).id
            self._database.delete_all_threads()
        if self._file_search:
            vectore_store_id = self._database.get_vector_store_id()
            if vectore_store_id:
                self._vector_store = self._client.beta.vector_stores.retrieve(
                    vector_store_id=vectore_store_id
                )
            else:
                uid = uuid.uuid4().hex
                self._vector_store = self._client.beta.vector_stores.create(
                    name=uid)
                self._database.save_vector_store_id(self._vector_store.id)
            self._client.beta.assistants.update(
                assistant_id=self._assistant_id,
                tool_resources={
                    "file_search": {"vector_store_ids": [self._vector_store.id]}
                },
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Perform any cleanup actions here
        pass

    async def _create_thread(self, user_id: str) -> str:
        thread_id = self._database.get_thread_id(user_id)

        if thread_id is None:
            thread = self._client.beta.threads.create()
            thread_id = thread.id
            self._database.save_thread_id(user_id, thread_id)
            if self._zep:
                try:
                    await self._zep.user.add(user_id=user_id)
                except Exception:
                    pass
                try:
                    await self._zep.memory.add_session(
                        user_id=user_id, session_id=user_id
                    )
                except Exception:
                    pass

        return thread_id

    async def _cancel_run(self, thread_id: str, run_id: str):
        try:
            self._client.beta.threads.runs.cancel(
                thread_id=thread_id, run_id=run_id)
        except Exception as e:
            print(f"Error cancelling run: {e}")

    async def _get_active_run(self, thread_id: str) -> Optional[str]:
        runs = self._client.beta.threads.runs.list(
            thread_id=thread_id, limit=1)
        for run in runs:
            if run.status in ["in_progress"]:
                return run.id
        return None

    async def _get_run_status(self, thread_id: str, run_id: str) -> str:
        run = self._client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )
        return run.status

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
        prompt: str = "Summarize the markdown table into a report, include important metrics and totals.",
        namespace: str = "global",
        model: Literal["gemini-2.0-flash",
                       "gemini-1.5-pro"] = "gemini-1.5-pro",
    ):
        """Upload and process a CSV file into the knowledge base with AI summarization.

        Args:
            file (BinaryIO): The CSV file to upload and process
            filename (str): The name of the CSV file
            prompt (str, optional): Custom prompt for summarization. Defaults to "Summarize the markdown table into a report, include important metrics and totals."
            namespace (str, optional): Knowledge base namespace. Defaults to "global".
            model (Literal["gemini-2.0-flash", "gemini-1.5-pro"], optional):
                Gemini model for summarization. Defaults to "gemini-1.5-pro"

        Example:
            ```python
            ai.upload_csv_file(
                file=open("data.csv", "rb"),
                filename="data.csv",
            )
            ```

        Note:
            - Converts CSV to Markdown table format
            - Uses Gemini AI to generate a summary
            - Stores summary in Pinecone knowledge base
            - Requires configured Pinecone index
            - Supports custom prompts for targeted summaries
        """
        csv_text = self.csv_to_text(file, filename)
        print(csv_text)
        document = self.summarize(csv_text, prompt, model)
        print(document)
        self.add_document_to_kb(document=document, namespace=namespace)

    def delete_vector_store_and_files(self):
        """Delete the OpenAI vector store and files.

        Example:
            ```python
            ai.delete_vector_store()
            ```

        Note:
            - Requires file_search=True in AI initialization
            - Deletes the vector store and all associated files
        """
        vector_store_id = self._database.get_vector_store_id()
        if vector_store_id:
            self._client.beta.vector_stores.delete(vector_store_id)
            self._database.delete_vector_store_id(vector_store_id)
            for file in self._database.files.find().to_list():
                self._client.files.delete(file["file_id"])
                self._database.delete_file(file["file_id"])

    def max_files(self) -> bool:
        """Check if the OpenAI vector store has reached its maximum file capacity.

        Returns:
            bool: True if file count is at maximum (10,000), False otherwise

        Example:
            ```python
            if ai.max_files():
                print("Vector store is full")
            else:
                print("Can still add more files")
            ```

        Note:
            - Requires file_search=True in AI initialization
            - OpenAI vector stores have a 10,000 file limit
            - Returns False if vector store is not configured
        """
        self._vector_store.file_counts.completed == 10000

    def file_count(self) -> int:
        """Get the total number of files processed in the OpenAI vector store.

        Returns:
            int: Number of successfully processed files in the vector store

        Example:
            ```python
            count = ai.file_count()
            print(f"Processed {count} files")
            # Returns: "Processed 5 files"
            ```

        Note:
            - Requires file_search=True in AI initialization
            - Only counts successfully processed files
            - Returns 0 if vector store is not configured
        """
        self._vector_store.file_counts.completed

    def add_file(
        self,
        filename: str,
        file_stream: bytes,
    ) -> Literal["in_progress", "completed", "cancelled", "failed"]:
        """Upload and process a file in the OpenAI vector store.

        Args:
            filename (str): Name of the file to upload
            file_stream (bytes): Raw bytes of the file to upload

        Returns:
            Literal["in_progress", "completed", "cancelled", "failed"]: Status of file processing

        Example:
            ```python
            with open('document.pdf', 'rb') as f:
                status = ai.add_file(f.filename, f.read())
                if status == "completed":
                    print("File processed successfully")
            ```

        Note:
            - Requires file_search=True in AI initialization
            - Files are vectorized for semantic search
            - Maximum file size: 512MB
            - Maximum 10,000 files per vector store
            - Processing may take a few seconds to minutes
        """
        vector_store_id = self._database.get_vector_store_id()
        file = self._client.files.create(
            file=(filename, file_stream), purpose="assistants"
        )
        file_batch = self._client.beta.vector_stores.files.create_and_poll(
            vector_store_id=vector_store_id, file_id=file.id
        )
        self._database.add_file(file.id)
        return file_batch.status

    def search_kb(self, query: str, namespace: str = "global", limit: int = 3) -> str:
        """Search Pinecone knowledge base using OpenAI embeddings.

        Args:
            query (str): Search query to find relevant documents
            namespace (str, optional): Namespace of the Pinecone to search. Defaults to "global".
            limit (int, optional): Maximum number of results to return. Defaults to 3.

        Returns:
            str: JSON string of matched documents or error message

        Example:
            ```python
            results = ai.search_kb("user123", "machine learning basics")
            # Returns: '["Document 1", "Document 2", ...]'
            ```

        Note:
            - Requires configured Pinecone index
            - Uses OpenAI embeddings for semantic search
            - Returns JSON-serialized Pinecone match metadata results
            - Returns error message string if search fails
            - Optionally reranks results using Cohere API
        """
        try:
            response = self._client.embeddings.create(
                input=query,
                model=self._openai_embedding_model,
            )
            search_results = self.kb.query(
                vector=response.data[0].embedding,
                top_k=10,
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
            if self._co:
                try:
                    response = self._co.rerank(
                        model=self._co_model,
                        query=query,
                        documents=docs,
                        top_n=limit,
                    )
                    reranked_docs = response.results
                    new_docs = []
                    for doc in reranked_docs:
                        new_docs.append(docs[doc.index])
                    return json.dumps(new_docs)
                except Exception:
                    return json.dumps(docs[:limit])
            else:
                return json.dumps(docs[:limit])
        except Exception as e:
            return f"Failed to search KB. Error: {e}"

    def add_document_to_kb(
        self,
        document: str,
        id: str = uuid.uuid4().hex,
        namespace: str = "global",
    ):
        """Add a document to the Pinecone knowledge base with OpenAI embeddings.

        Args:
            document (str): Document to add to the knowledge base
            id (str, optional): Unique identifier for the document. Defaults to random UUID.
            namespace (str): Namespace of the Pinecone index to search. Defaults to "global".

        Example:
            ```python
            ai.add_document_to_kb("user123 has 4 cats")
            ```

        Note:
            - Requires Pinecone index to be configured
            - Uses OpenAI embeddings API
        """
        response = self._client.embeddings.create(
            input=document,
            model=self._openai_embedding_model,
        )
        self.kb.upsert(
            vectors=[
                {
                    "id": id,
                    "values": response.data[0].embedding,
                }
            ],
            namespace=namespace,
        )
        self._database.add_document_to_kb(id, namespace, document)

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

    # check time tool - has to be sync
    def check_time(self) -> str:
        """Get current UTC time formatted as a string.

        Returns:
            str: Current UTC time in format 'YYYY-MM-DD HH:MM:SS UTC'

        Example:
            ```python
            time = ai.check_time()
            # Returns: "2024-02-13 15:30:45 UTC"
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
            Always returns time in UTC timezone for consistency.
        """
        return datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )

    # search facts tool - has to be sync
    def search_facts(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> str:
        """Search stored conversation facts using Zep memory integration.

        Args:
            user_id (str): Unique identifier for the user
            query (str): Search query to find relevant facts
            limit (int, optional): Maximum number of facts to return. Defaults to 10.

        Returns:
            str: JSON string of matched facts or error message

        Example:
            ```python
            facts = ai.search_facts(
                user_id="user123",
                query="How many cats do I have?"
            )
            # Returns: [{"fact": "user123 has 4 cats", "timestamp": "2022-01-01T12:00:00Z"}]
            ```

        Note:
            Requires Zep integration to be configured with valid API key.
            This is a synchronous tool method required for OpenAI function calling.
        """
        if self._sync_zep:
            try:
                facts = []
                results = self._sync_zep.memory.search_sessions(
                    user_id=user_id,
                    session_ids=[user_id],
                    text=query,
                    limit=limit,
                )
                for result in results.results:
                    fact = result.fact.fact
                    timestamp = result.fact.created_at
                    facts.append({"fact": fact, "timestamp": timestamp})
                return json.dumps(facts)
            except Exception as e:
                return f"Failed to search facts. Error: {e}"
        else:
            return "Zep integration not configured."

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

    # reason tool - has to be sync
    def reason(
        self,
        user_id: str,
        query: str,
        prompt: str = "You combine the data with your reasoning to answer the query.",
        use_perplexity: bool = True,
        use_grok: bool = True,
        use_facts: bool = True,
        use_kb: bool = True,
        perplexity_model: Literal[
            "sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning"
        ] = "sonar",
        openai_model: Literal["o1", "o3-mini"] = "o3-mini",
        grok_model: Literal["grok-2-latest"] = "grok-2-latest",
        namespace: str = "global",
    ) -> str:
        """Combine multiple data sources with AI reasoning to answer queries.

        Args:
            user_id (str): Unique identifier for the user
            query (str): The question or query to reason about
            prompt (str, optional): Prompt for reasoning. Defaults to "You combine the data with your reasoning to answer the query."
            use_perplexity (bool, optional): Include Perplexity search results. Defaults to True
            use_grok (bool, optional): Include X/Twitter search results. Defaults to True
            use_facts (bool, optional): Include stored conversation facts. Defaults to True
            use_kb (bool, optional): Include Pinecone knowledge base search results. Defaults to True
            perplexity_model (Literal, optional): Perplexity model to use. Defaults to "sonar"
            openai_model (Literal, optional): OpenAI model for reasoning. Defaults to "o3-mini"
            grok_model (Literal, optional): Grok model for X search. Defaults to "grok-beta"
            namespace (str): Namespace of the Pinecone index to search. Defaults to "global"

        Returns:
            str: Reasoned response combining all enabled data sources or error message

        Example:
            ```python
            result = ai.reason(
                user_id="user123",
                query="What are the latest AI trends?",
            )
            # Returns: "Based on multiple sources: [comprehensive answer]"
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
            Requires configuration of relevant API keys for enabled data sources.
            Will gracefully handle missing or failed data sources.
        """
        try:
            if use_kb:
                try:
                    kb_results = self.search_kb(query, namespace)
                except Exception:
                    kb_results = ""
            else:
                kb_results = ""
            if use_facts:
                try:
                    facts = self.search_facts(user_id, query)
                except Exception:
                    facts = ""
            else:
                facts = ""
            if use_perplexity:
                try:
                    search_results = self.search_internet(
                        query, perplexity_model)
                except Exception:
                    search_results = ""
            else:
                search_results = ""
            if use_grok:
                try:
                    x_search_results = self.search_x(query, grok_model)
                except Exception:
                    x_search_results = ""
            else:
                x_search_results = ""

            response = self._client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}, Facts: {facts}, KB Results: {kb_results}, Internet Search Results: {search_results}, X Search Results: {x_search_results}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to reason. Error: {e}"

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
            await self.delete_assistant_thread(user_id)
        except Exception:
            pass
        try:
            self._database.clear_user_history(user_id)
        except Exception:
            pass
        try:
            await self.delete_facts(user_id)
        except Exception:
            pass

    async def delete_assistant_thread(self, user_id: str):
        """Delete stored conversation thread for a user on OpenAI.

        Example:
            ```python
            await ai.delete_assistant_thread("user123")
            # Deletes the assistant conversation thread for a user
            ```
        """
        thread_id = self._database.get_thread_id(user_id)
        await self._client.beta.threads.delete(thread_id=thread_id)

    async def delete_facts(self, user_id: str):
        """Delete stored conversation facts for a specific user from Zep memory.

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
            user_id (str): Unique identifier for the user/conversation
            user_text (str): Text input from user to process

        Returns:
            AsyncGenerator[str, None]: Stream of response text chunks

        Example:
            ```python
            async for chunk in ai.text("user123", "What is machine learning?"):
                print(chunk, end="")  # Prints response as it streams
            ```

        Note:
            - Maintains conversation thread using OpenAI's thread system
            - Stores messages in configured database (MongoDB/SQLite)
            - Integrates with Zep memory if configured
            - Handles concurrent runs by canceling active ones
            - Streams responses for real-time interaction
        """
        self._accumulated_value_queue = asyncio.Queue()

        thread_id = self._database.get_thread_id(user_id)

        if thread_id is None:
            thread_id = await self._create_thread(user_id)

        self._current_thread_id = thread_id

        # Check for active runs and cancel if necessary
        active_run_id = await self._get_active_run(thread_id)
        if active_run_id:
            await self._cancel_run(thread_id, active_run_id)
            while await self._get_run_status(thread_id, active_run_id) != "cancelled":
                await asyncio.sleep(0.1)

        # Create a message in the thread
        self._client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_text,
        )
        event_handler = EventHandler(self._tool_handlers, self)

        async def stream_processor():
            with self._client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=self._assistant_id,
                event_handler=event_handler,
            ) as stream:
                stream.until_done()

        # Start the stream processor in a separate task
        asyncio.create_task(stream_processor())

        # Yield values from the queue as they become available
        full_response = ""
        while True:
            try:
                value = await asyncio.wait_for(
                    self._accumulated_value_queue.get(), timeout=0.1
                )
                if value is not None:
                    full_response += value
                    yield value
            except asyncio.TimeoutError:
                if self._accumulated_value_queue.empty():
                    break

        # Save the message to the database
        metadata = {
            "user_id": user_id,
            "message": user_text,
            "response": full_response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }

        self._database.save_message(user_id, metadata)
        if self._zep:
            messages = [
                Message(
                    role="user",
                    role_type="user",
                    content=user_text,
                ),
                Message(
                    role="assistant",
                    role_type="assistant",
                    content=full_response,
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

        # Reset the queue for each new conversation
        self._accumulated_value_queue = asyncio.Queue()

        thread_id = self._database.get_thread_id(user_id)

        if thread_id is None:
            thread_id = await self._create_thread(user_id)

        self._current_thread_id = thread_id
        transcript = await self._listen(audio_bytes, input_format)
        event_handler = EventHandler(self._tool_handlers, self)
        self._client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=transcript,
        )

        async def stream_processor():
            with self._client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=self._assistant_id,
                event_handler=event_handler,
            ) as stream:
                stream.until_done()

        # Start the stream processor in a separate task
        asyncio.create_task(stream_processor())

        # Collect the full response
        full_response = ""
        while True:
            try:
                value = await asyncio.wait_for(
                    self._accumulated_value_queue.get(), timeout=0.1
                )
                if value is not None:
                    full_response += value
            except asyncio.TimeoutError:
                if self._accumulated_value_queue.empty():
                    break

        metadata = {
            "user_id": user_id,
            "message": transcript,
            "response": full_response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }

        self._database.save_message(user_id, metadata)

        if self._zep:
            messages = [
                Message(
                    role="user",
                    role_type="user",
                    content=transcript,
                ),
                Message(
                    role="assistant",
                    role_type="assistant",
                    content=full_response,
                ),
            ]
            await self._zep.memory.add(session_id=user_id, messages=messages)

        # Generate and stream the audio response
        with self._client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=full_response,
            response_format=response_format,
        ) as response:
            for chunk in response.iter_bytes(1024):
                yield chunk

    def _handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name in self._tool_handlers:
                handler = self._tool_handlers[tool.function.name]
                inputs = json.loads(tool.function.arguments)
                output = handler(**inputs)
                tool_outputs.append(
                    {"tool_call_id": tool.id, "output": output})

        self._submit_tool_outputs(tool_outputs, run_id)

    def _submit_tool_outputs(self, tool_outputs, run_id):
        with self._client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self._current_thread_id, run_id=run_id, tool_outputs=tool_outputs
        ) as stream:
            for text in stream.text_deltas:
                asyncio.create_task(self._accumulated_value_queue.put(text))

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
        self._tool_handlers[func.__name__] = func
        return func


tool = AI.add_tool
