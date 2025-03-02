import asyncio
import datetime
import traceback
import ntplib
import json
from typing import AsyncGenerator, List, Literal, Dict, Any, Callable
import uuid
import pandas as pd
from pydantic import BaseModel, Field
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


# Define Pydantic models for structured output
class ImprovementArea(BaseModel):
    area: str = Field(...,
                      description="Area name (e.g., 'Accuracy', 'Completeness')")
    issue: str = Field(..., description="Specific issue identified")
    recommendation: str = Field(...,
                                description="Specific actionable improvement")


class CritiqueFeedback(BaseModel):
    strengths: List[str] = Field(
        default_factory=list, description="List of strengths in the response"
    )
    improvement_areas: List[ImprovementArea] = Field(
        default_factory=list, description="Areas needing improvement"
    )
    overall_score: float = Field(..., description="Score between 0.0 and 1.0")
    priority: Literal["low", "medium", "high"] = Field(
        ..., description="Priority level for improvements"
    )


class DocumentModel(BaseModel):
    id: str
    text: str


class MongoDatabase:
    def __init__(self, db_url: str, db_name: str):
        self._client = MongoClient(db_url)
        self.db = self._client[db_name]
        self.messages = self.db["messages"]
        self.kb = self.db["kb"]
        self.jobs = self.db["jobs"]

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
        research_model: str = "gpt-4o-mini",
        enable_internet_search: bool = True,
        default_timezone: str = "UTC",
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
            research_model (str, optional): Model for research. Defaults to "gpt-4o-mini"
            enable_internet_search (bool, optional): Enable internet search tools. Defaults to True
            default_timezone (str, optional): Default timezone for time awareness. Defaults to "UTC"
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
        self._research_model = research_model
        self._tools = []
        self._job_processor_task = None
        self._default_timezone = default_timezone

        # Automatically add internet search tool if API key is provided and feature is enabled
        if perplexity_api_key and enable_internet_search:
            # Use the add_tool decorator functionality directly
            search_internet_tool = {
                "type": "function",
                "function": {
                    "name": "search_internet",
                    "description": "Search the internet using Perplexity AI API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string",
                            },
                            "model": {
                                "type": "string",
                                "description": "Perplexity model to use",
                                "enum": [
                                    "sonar",
                                    "sonar-pro",
                                    "sonar-reasoning-pro",
                                    "sonar-reasoning",
                                ],
                                "default": "sonar",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            self._tools.append(search_internet_tool)
            print("Internet search capability added as default tool")

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

    def check_time(self, timezone: str = None) -> str:
        """Get current time in requested timezone as a string.

        Args:
            timezone (str, optional): Timezone to convert the time to (e.g., "America/New_York").
              If None, uses the agent's default timezone.

        Returns:
            str: Current time in the requested timezone in format 'YYYY-MM-DD HH:MM:SS'
        """
        # Use provided timezone or fall back to agent default
        timezone = timezone or self._default_timezone or "UTC"

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

                # Format exactly as the test expects
                return f"current time in {timezone} is {formatted_time}"

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

    def make_time_aware(self, default_timezone="UTC"):
        """Make the agent time-aware by adding time checking capability."""
        # Add time awareness to instructions with explicit formatting guidance
        time_instructions = f"""
        IMPORTANT: You are time-aware. The current date is {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")}.
        
        TIME RESPONSE RULES:
        1. When asked about the current time, ONLY use the check_time tool and respond with EXACTLY what it returns
        2. NEVER add UTC time when the check_time tool returns local time
        3. NEVER convert between timezones unless explicitly requested
        4. NEVER mention timezone offsets (like "X hours behind UTC") unless explicitly asked
        5. Local time is the ONLY time that should be mentioned in your response
        
        Default timezone: {default_timezone} (use this when user's timezone is unknown)
        """
        self._instructions = self._instructions + "\n\n" + time_instructions

        self._default_timezone = default_timezone

        # Ensure the check_time tool is registered (in case it was removed)
        existing_tools = [t["function"]["name"] for t in self._tools]
        if "check_time" not in existing_tools:
            # Get method reference
            check_time_func = self.check_time
            # Re-register it using our add_tool decorator
            self.add_tool(check_time_func)

        return self

    async def research_and_learn(self, topic: str) -> str:
        """Research a topic and add findings to collective memory.

        Args:
            topic: The topic to research and learn about

        Returns:
            Summary of what was learned
        """
        try:
            # First, search the internet for information
            search_results = await self.search_internet(
                f"comprehensive information about {topic}"
            )

            # Extract structured knowledge
            prompt = f"""
            Based on these search results about "{topic}", extract 3-5 factual insights 
            worth adding to our collective knowledge.
            
            Search results:
            {search_results}
            
            Format each insight as a JSON object with:
            1. "fact": The factual information
            2. "relevance": Short explanation of why this is generally useful
            
            Return ONLY a valid JSON array. Example:
            [
                {{"fact": "Topic X has property Y", "relevance": "Important for understanding Z"}}
            ]
            """

            response = self._client.chat.completions.create(
                model=self._research_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract factual knowledge from research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            insights = json.loads(response.choices[0].message.content)

            # Add to collective memory via the swarm
            if hasattr(self, "_swarm") and self._swarm and insights:
                conversation = {
                    "message": f"Research on {topic}",
                    "response": json.dumps(insights),
                    "user_id": "system_explorer",
                }
                await self._swarm.extract_and_store_insights(
                    "system_explorer", conversation
                )

                # Return a summary of what was learned
                return f"✅ Added {len(insights)} new insights about '{topic}' to collective memory."

            return "⚠️ Could not add insights to collective memory."

        except Exception as e:
            return f"❌ Error researching topic: {str(e)}"

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

    async def text(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
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
        # Store current user ID for task scheduling context
        self._current_user_id = user_id

        # Store timezone with user ID for persistence
        if timezone:
            if not hasattr(self, "_user_timezones"):
                self._user_timezones = {}
            self._user_timezones[user_id] = timezone

        # Set current timezone for this session
        self._current_timezone = (
            timezone
            if timezone
            else self._user_timezones.get(user_id, self._default_timezone)
        )

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


class Swarm:
    """An AI Agent Swarm that coordinates specialized AI agents with handoff capabilities."""

    def __init__(
        self,
        database: MongoDatabase,
        swarm_directive: str = None,
        router_model: str = "gpt-4o-mini",
        insight_model: str = "gpt-4o-mini",
        enable_collective_memory: bool = True,
        enable_autonomous_learning: bool = True,
        default_timezone: str = "UTC",
        enable_critic: bool = True,  # New parameter
    ):
        """Initialize the multi-agent system with a shared database.

        Args:
            database (MongoDatabase): Shared MongoDB database instance
            swarm_directive (str, optional): Core directive/mission that governs all agents. Defaults to None.
            router_model (str, optional): Model to use for routing decisions. Defaults to "gpt-4o-mini".
            insight_model (str, optional): Model to extract collective insights. Defaults to "gpt-4o-mini".
            enable_collective_memory (bool, optional): Whether to enable collective memory. Defaults to True.
            enable_autonomous_learning (bool, optional): Whether to enable autonomous learning. Defaults to True.
            default_timezone (str, optional): Default timezone for time-awareness. Defaults to "UTC".
            enable_critic (bool, optional): Whether to enable the critic system. Defaults to True.
            critique_frequency (float, optional): Fraction of interactions to analyze. Defaults to 0.1.
        """
        self.agents = {}  # name -> AI instance
        self.specializations = {}  # name -> description
        self.database = database
        self.router_model = router_model
        self.insight_model = insight_model
        self.enable_collective_memory = enable_collective_memory
        self.enable_autonomous_learning = enable_autonomous_learning
        self.default_timezone = default_timezone
        self.enable_critic = enable_critic

        # Store swarm directive
        self.swarm_directive = (
            swarm_directive
            or """
        You are part of an agent swarm that works together to serve users effectively.
        Your goals are to provide accurate, helpful responses while collaborating with other agents.
        """
        )

        self.formatted_directive = f"""
            ┌─────────────── SWARM DIRECTIVE ───────────────┐
            {self.swarm_directive}
            └─────────────────────────────────────────────┘
        """

        # Initialize critic if enabled
        if enable_critic:
            self.critic = Critic(
                self,
                critique_model=insight_model,
            )

        # Ensure handoffs collection exists
        if "handoffs" not in self.database.db.list_collection_names():
            self.database.db.create_collection("handoffs")
        self.handoffs = self.database.db["handoffs"]

        # Create collective memory collection
        if enable_collective_memory:
            if "collective_memory" not in self.database.db.list_collection_names():
                self.database.db.create_collection("collective_memory")
            self.collective_memory = self.database.db["collective_memory"]

            # Create text index for MongoDB text search
            try:
                self.collective_memory.create_index(
                    [("fact", "text"), ("relevance", "text")]
                )
                print("Created text search index for collective memory")
            except Exception as e:
                print(f"Warning: Text index creation might have failed: {e}")
        else:
            print("Collective memory feature is disabled")

        print(
            f"MultiAgentSystem initialized with router model: {router_model}")

        # Update the extract_and_store_insights method in Swarm class

    async def extract_and_store_insights(
        self, user_id: str, conversation: dict
    ) -> None:
        """Extract and store insights with hybrid vector/text search capabilities."""
        # Get first agent to use its OpenAI client
        if not self.agents:
            return

        first_agent = next(iter(self.agents.values()))

        # Create the prompt to extract insights
        prompt = f"""
        Review this conversation and extract 0-3 IMPORTANT factual insights worth remembering for future users.
        Only extract FACTUAL information that would be valuable across multiple conversations.
        Do NOT include opinions, personal preferences, or user-specific details.
        
        Conversation:
        User: {conversation.get('message', '')}
        Assistant: {conversation.get('response', '')}
        
        Format each insight as a JSON object with:
        1. "fact": The factual information
        2. "relevance": Short explanation of why this is generally useful
        
        Return ONLY a valid JSON array, even if empty. Example:
        [
            {{"fact": "Solana processes 65,000 TPS", "relevance": "Important performance metric for blockchain comparisons"}}
        ]
        """

        # Extract insights using AI
        try:
            response = first_agent._client.chat.completions.create(
                model=self.insight_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract important factual insights from conversations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            insights_text = response.choices[0].message.content
            insights = json.loads(insights_text)

            # Store in MongoDB (keeps all metadata and text)
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            mongo_records = []

            for insight in insights:
                record_id = str(uuid.uuid4())
                insight["_id"] = record_id
                insight["timestamp"] = timestamp
                insight["source_user_id"] = user_id
                mongo_records.append(insight)

            if mongo_records:
                for record in mongo_records:
                    self.collective_memory.insert_one(record)

            # Also store in Pinecone for semantic search if available
            if (
                insights
                and hasattr(first_agent, "_pinecone")
                and first_agent._pinecone
                and first_agent.kb
            ):
                try:
                    # Generate embeddings
                    texts = [
                        f"{insight['fact']}: {insight['relevance']}"
                        for insight in insights
                    ]
                    embeddings = first_agent._pinecone.inference.embed(
                        model=first_agent._pinecone_embedding_model, inputs=texts
                    )

                    # Create vectors for Pinecone
                    vectors = []
                    for insight, embedding in zip(insights, embeddings):
                        vectors.append(
                            {
                                "id": insight["_id"],
                                "values": embedding.values,
                                "metadata": {
                                    "fact": insight["fact"],
                                    "relevance": insight["relevance"],
                                    "timestamp": str(timestamp),
                                    "source_user_id": user_id,
                                },
                            }
                        )

                    # Store in Pinecone
                    first_agent.kb.upsert(
                        vectors=vectors, namespace="collective_memory"
                    )
                    print(
                        f"Stored {len(insights)} insights in both MongoDB and Pinecone"
                    )
                except Exception as e:
                    print(f"Error storing insights in Pinecone: {e}")
            else:
                print(f"Stored {len(insights)} insights in MongoDB only")

        except Exception as e:
            print(f"Failed to extract insights: {str(e)}")

    def search_collective_memory(self, query: str, limit: int = 5) -> str:
        """Search the collective memory using a hybrid approach.

        First tries semantic vector search through Pinecone, then falls back to
        MongoDB text search, and finally to recency-based search as needed.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            Formatted string with relevant insights
        """
        try:
            if not self.enable_collective_memory:
                return "Collective memory feature is disabled."

            results = []
            search_method = "recency"  # Default method if others fail

            # Try semantic search with Pinecone first
            if self.agents:
                first_agent = next(iter(self.agents.values()))
                if (
                    hasattr(first_agent, "_pinecone")
                    and first_agent._pinecone
                    and first_agent.kb
                ):
                    try:
                        # Generate embedding for query
                        embedding = first_agent._pinecone.inference.embed(
                            model=first_agent._pinecone_embedding_model, inputs=[
                                query]
                        )

                        # Search Pinecone
                        pinecone_results = first_agent.kb.query(
                            vector=embedding[0].values,
                            top_k=limit * 2,  # Get more results to allow for filtering
                            include_metadata=True,
                            namespace="collective_memory",
                        )

                        # Extract results from Pinecone
                        if pinecone_results.matches:
                            for match in pinecone_results.matches:
                                if hasattr(match, "metadata") and match.metadata:
                                    results.append(
                                        {
                                            "fact": match.metadata.get(
                                                "fact", "Unknown fact"
                                            ),
                                            "relevance": match.metadata.get(
                                                "relevance", ""
                                            ),
                                            "score": match.score,
                                        }
                                    )

                            # Get top results
                            results = sorted(
                                results, key=lambda x: x.get("score", 0), reverse=True
                            )[:limit]
                            search_method = "semantic"
                    except Exception as e:
                        print(f"Pinecone search error: {e}")

            # Fall back to MongoDB keyword search if needed
            if not results:
                try:
                    # First try text search if we have the index
                    mongo_results = list(
                        self.collective_memory.find(
                            {"$text": {"$search": query}},
                            {"score": {"$meta": "textScore"}},
                        )
                        .sort([("score", {"$meta": "textScore"})])
                        .limit(limit)
                    )

                    if mongo_results:
                        results = mongo_results
                        search_method = "keyword"
                    else:
                        # Fall back to most recent insights
                        results = list(
                            self.collective_memory.find()
                            .sort("timestamp", -1)
                            .limit(limit)
                        )
                        search_method = "recency"
                except Exception as e:
                    print(f"MongoDB search error: {e}")
                    # Final fallback - just get most recent
                    results = list(
                        self.collective_memory.find().sort("timestamp", -1).limit(limit)
                    )

            # Format the results
            if not results:
                return "No collective knowledge available."

            formatted = [
                f"## Relevant Collective Knowledge (using {search_method} search)"
            ]
            for insight in results:
                formatted.append(
                    f"- **{insight.get('fact')}** _{insight.get('relevance', '')}_"
                )

            return "\n".join(formatted)

        except Exception as e:
            print(f"Error searching collective memory: {str(e)}")
            return "Error retrieving collective knowledge."

    def register(self, name: str, agent: AI, specialization: str):
        """Register a specialized agent with the multi-agent system."""
        # Make agent time-aware first
        agent.make_time_aware(self.default_timezone)

        # Apply swarm directive to the agent
        agent._instructions = f"{self.formatted_directive}\n\n{agent._instructions}"

        # Add the agent to the system first
        self.agents[name] = agent
        self.specializations[name] = specialization

        # Add collective memory tool to the agent
        @agent.add_tool
        def query_collective_knowledge(query: str) -> str:
            """Query the swarm's collective knowledge from all users.

            Args:
                query (str): The search query to look for in collective knowledge

            Returns:
                str: Relevant insights from the swarm's collective memory
            """
            return self.search_collective_memory(query)

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

            # First remove any existing handoff tool if present
            agent._tools = [
                t for t in agent._tools if t["function"]["name"] != "request_handoff"
            ]

            # Create handoff tool with explicit naming requirements
            def create_handoff_tool(current_agent_name, available_targets_list):
                def request_handoff(target_agent: str, reason: str) -> str:
                    """Request an immediate handoff to another specialized agent.
                    This is an INTERNAL SYSTEM TOOL. The user will NOT see your reasoning about the handoff.
                    Use this tool IMMEDIATELY when a query is outside your expertise.

                    Args:
                        target_agent (str): Name of agent to transfer to. MUST be one of: {', '.join(available_targets_list)}.
                          DO NOT INVENT NEW NAMES OR VARIATIONS. Use EXACTLY one of these names.
                        reason (str): Brief explanation of why this question requires the specialist

                    Returns:
                        str: Empty string - the handoff is handled internally
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

                    # Set handoff info in the agent instance for processing
                    agent._handoff_info = {
                        "target": target_agent, "reason": reason}

                    # Return empty string - the actual handoff happens in the process method
                    return ""

                return request_handoff

            # Use the factory to create a properly-bound tool function
            handoff_tool = create_handoff_tool(agent_name, available_targets)

            # Initialize handoff info attribute
            agent._handoff_info = None

            # Add the updated handoff tool with proper closure
            agent.add_tool(handoff_tool)

            # Add critical handoff instructions to the agent's base instructions
            handoff_examples = "\n".join(
                [
                    f"  - `{name}` ({self.specializations[name][:40]}...)"
                    for name in available_targets
                ]
            )
            handoff_instructions = f"""
            STRICT HANDOFF GUIDANCE:
            1. You must use ONLY the EXACT agent names listed below for handoffs:
               {handoff_examples}
               
            2. DO NOT INVENT, MODIFY, OR CREATE NEW AGENT NAMES like "Smart Contract Developer" or "Technical Expert"
            
            3. For technical implementation questions, use "developer" (not variations like "developer expert" or "tech specialist")
            
            4. ONLY these EXACT agent names will work for handoffs: {', '.join(available_targets)}
            """

            # Update agent instructions with handoff guidance
            agent._instructions = agent._instructions + "\n\n" + handoff_instructions

            print(
                f"Updated handoff capabilities for {agent_name} with targets: {available_targets}"
            )

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle handoffs.

        Args:
            user_id (str): Unique user identifier
            user_text (str): User's text input
            timezone (str, optional): User-specific timezone
        """
        try:
            # Handle special commands
            if user_text.strip().lower().startswith("!memory "):
                query = user_text[8:].strip()
                yield self.search_collective_memory(query)
                return

            # Check for registered agents
            if not self.agents:
                yield "Error: No agents are registered with the system. Please register at least one agent first."
                return

            # Get initial routing and agent
            first_agent = next(iter(self.agents.values()))
            agent_name = await self._get_routing_decision(first_agent, user_text)
            current_agent = self.agents[agent_name]
            print(f"Starting conversation with agent: {agent_name}")

            # Reset handoff info
            current_agent._handoff_info = None

            # Response tracking
            final_response = ""
            confidence_score = 1.0  # Default high confidence

            # Process response stream
            async for chunk in self._stream_response(
                user_id, user_text, current_agent, timezone
            ):
                yield chunk
                final_response += chunk

                # Detect confidence signals in response
                lower_chunk = chunk.lower()
                if any(
                    phrase in lower_chunk
                    for phrase in [
                        "i'm not sure",
                        "uncertain",
                        "i don't know",
                        "limited information",
                    ]
                ):
                    confidence_score = 0.4
                elif any(
                    phrase in lower_chunk
                    for phrase in ["might be", "possibly", "perhaps"]
                ):
                    confidence_score = 0.7

            # Post-processing: learn from conversation
            conversation = {
                "user_id": user_id,
                "message": user_text,
                "response": final_response,
            }

            # Run post-processing tasks concurrently
            tasks = []

            # Add collective memory task if enabled
            if self.enable_collective_memory:
                tasks.append(self.extract_and_store_insights(
                    user_id, conversation))

            # Add autonomous learning task if enabled and confidence is low
            if (
                self.enable_autonomous_learning
                and hasattr(self, "knowledge_explorer")
                and confidence_score < 0.8
            ):
                tasks.append(
                    self.knowledge_explorer.analyze_interaction(
                        user_id, user_text, final_response, confidence_score
                    )
                )

            # Run all post-processing tasks concurrently
            if tasks:
                # Don't block - run asynchronously
                asyncio.create_task(self._run_post_processing_tasks(tasks))

        except Exception as e:
            print(f"Error in multi-agent processing: {str(e)}")
            print(traceback.format_exc())
            yield "\n\nI apologize for the technical difficulty.\n\n"

    async def _stream_response(
        self, user_id, user_text, current_agent, timezone=None
    ) -> AsyncGenerator[str, None]:
        """Stream response from an agent, handling potential handoffs."""
        handoff_detected = False
        response_started = False
        full_response = ""

        # Get recent feedback for this agent to improve the response
        agent_name = current_agent.__class__.__name__
        recent_feedback = []

        if self.enable_critic and hasattr(self, "critic"):
            try:
                # Get the most recent feedback for this agent
                feedback_records = list(
                    self.critic.feedback_collection.find(
                        {"agent_name": agent_name})
                    .sort("timestamp", -1)
                    .limit(3)
                )

                if feedback_records:
                    # Extract specific improvement suggestions
                    for record in feedback_records:
                        recent_feedback.append(
                            f"- Improve {record.get('improvement_area')}: {record.get('recommendation')}"
                        )
            except Exception as e:
                print(f"Error getting recent feedback: {e}")

        # Augment user text with feedback instructions if available
        augmented_instruction = user_text
        if recent_feedback:
            feedback_text = "\n".join(recent_feedback)
            augmented_instruction = f"""
            Answer this question: {user_text}
        
        IMPORTANT - Apply these specific improvements from previous feedback:
        {feedback_text}
        """
        print(f"Applying feedback to improve response: {feedback_text}")

        async for chunk in current_agent.text(user_id, augmented_instruction, timezone):
            # Accumulate the full response for critic analysis
            full_response += chunk

            # Check for handoff after each chunk
            if current_agent._handoff_info and not handoff_detected:
                handoff_detected = True
                target_name = current_agent._handoff_info["target"]
                target_agent = self.agents[target_name]
                reason = current_agent._handoff_info["reason"]

                # Record handoff without waiting
                asyncio.create_task(
                    self._record_handoff(
                        user_id, current_agent, target_name, reason, user_text
                    )
                )

                # Add separator if needed
                if response_started:
                    yield "\n\n---\n\n"

                # Pass to target agent with comprehensive instructions
                handoff_query = f"""
                Answer this ENTIRE question completely from scratch:
                {user_text}
                
                IMPORTANT INSTRUCTIONS:
                1. Address ALL aspects of the question comprehensively
                2. Include both explanations AND implementations as needed
                3. Do not mention any handoff or that you're continuing from another agent
                4. Consider any relevant context from previous conversation
                """

                # Stream from target agent
                async for new_chunk in target_agent.text(user_id, handoff_query):
                    yield new_chunk
                    await asyncio.sleep(0)  # Force immediate delivery
                return

            # Regular response if no handoff detected
            if not handoff_detected:
                response_started = True
                yield chunk
                await asyncio.sleep(0)  # Force immediate delivery

        # After full response is delivered, invoke critic (if enabled)
        if self.enable_critic and hasattr(self, "critic"):
            # Don't block - run asynchronously
            asyncio.create_task(
                self.critic.analyze_interaction(
                    agent_name=current_agent.__class__.__name__,
                    user_query=user_text,
                    response=full_response,
                )
            )

    async def _run_post_processing_tasks(self, tasks):
        """Run multiple post-processing tasks concurrently."""
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error in post-processing tasks: {e}")

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


class Critic:
    """System that evaluates agent responses and suggests improvements."""

    def __init__(self, swarm, critique_model="gpt-4o-mini"):
        """Initialize the critic system.

        Args:
            swarm: The agent swarm to monitor
            critique_model: Model to use for evaluations
        """
        self.swarm = swarm
        self.critique_model = critique_model
        self.feedback_collection = swarm.database.db["agent_feedback"]

        # Create index for feedback collection
        if "agent_feedback" not in swarm.database.db.list_collection_names():
            swarm.database.db.create_collection("agent_feedback")
            self.feedback_collection.create_index([("agent_name", 1)])
            self.feedback_collection.create_index([("improvement_area", 1)])

    async def analyze_interaction(
        self,
        agent_name,
        user_query,
        response,
    ):
        """Analyze an agent interaction and provide improvement feedback."""
        # Get first agent's client for analysis
        first_agent = next(iter(self.swarm.agents.values()))

        prompt = f"""
        Analyze this agent interaction to identify specific improvements.
        
        INTERACTION:
        User query: {user_query}
        Agent response: {response}
        
        Provide feedback on accuracy, completeness, clarity, efficiency, and tone.
        """

        try:
            # Parse the response
            completion = first_agent._client.beta.chat.completions.parse(
                model=self.critique_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful critic evaluating AI responses.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=CritiqueFeedback,
                temperature=0.2,
            )

            # Extract the Pydantic model
            feedback = completion.choices[0].message.parsed

            # Manual validation - ensure score is between 0 and 1
            if feedback.overall_score < 0:
                feedback.overall_score = 0.0
            elif feedback.overall_score > 1:
                feedback.overall_score = 1.0

            # Store feedback in database
            for area in feedback.improvement_areas:
                self.feedback_collection.insert_one(
                    {
                        "agent_name": agent_name,
                        "user_query": user_query,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                        "improvement_area": area.area,
                        "issue": area.issue,
                        "recommendation": area.recommendation,
                        "overall_score": feedback.overall_score,
                        "priority": feedback.priority,
                    }
                )

            # If high priority feedback, schedule immediate learning task
            if feedback.priority == "high" and feedback.improvement_areas:
                top_issue = feedback.improvement_areas[0]
                await self.schedule_improvement_task(agent_name, top_issue)

            return feedback

        except Exception as e:
            print(f"Error in critic analysis: {str(e)}")
            return None

    async def schedule_improvement_task(self, agent_name, issue):
        """Execute improvement task immediately."""
        if agent_name in self.swarm.agents:
            agent = self.swarm.agents[agent_name]

            # Create topic for improvement
            topic = (
                f"How to improve {issue['area'].lower()} in responses: {issue['issue']}"
            )

            # Execute research directly
            result = await agent.research_and_learn(topic)

            print(
                f"📝 Executed improvement task for {agent_name}: {issue['area']}")
            return result

    def get_agent_feedback(self, agent_name=None, limit=10):
        """Get recent feedback for an agent or all agents."""
        query = {"agent_name": agent_name} if agent_name else {}
        feedback = list(
            self.feedback_collection.find(query).sort(
                "timestamp", -1).limit(limit)
        )
        return feedback

    def get_improvement_trends(self):
        """Get trends in improvement areas across all agents."""
        pipeline = [
            {
                "$group": {
                    "_id": "$improvement_area",
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$overall_score"},
                }
            },
            {"$sort": {"count": -1}},
        ]
        return list(self.feedback_collection.aggregate(pipeline))
