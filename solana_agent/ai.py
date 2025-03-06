import asyncio
import datetime
import random
import re
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


# Add this to the top of the file with other Pydantic models
class TicketResolution(BaseModel):
    status: Literal["resolved", "needs_followup", "cannot_determine"] = Field(
        ..., description="Resolution status of the ticket"
    )
    confidence: float = Field(
        ..., description="Confidence score for the resolution decision (0.0-1.0)"
    )
    reasoning: str = Field(
        ..., description="Brief explanation for the resolution decision"
    )
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested follow-up actions if needed"
    )


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


class MemoryInsight(BaseModel):
    fact: str = Field(...,
                      description="The factual information worth remembering")
    relevance: str = Field(
        ..., description="Short explanation of why this fact is generally useful"
    )


class NPSResponse(BaseModel):
    score: int = Field(..., ge=0, le=10, description="NPS score (0-10)")
    feedback: str = Field("", description="Optional feedback comment")
    improvement_suggestions: str = Field(
        "", description="Suggestions for improvement")


class CollectiveMemoryResponse(BaseModel):
    insights: List[MemoryInsight] = Field(
        default_factory=list,
        description="List of factual insights extracted from the conversation",
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
        self._client = OpenAI(api_key=openai_api_key)
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

        # Automatically add knowledge base search tool if Pinecone is configured
        if pinecone_api_key and pinecone_index_name:
            search_kb_tool = {
                "type": "function",
                "function": {
                    "name": "search_kb",
                    "description": "Search the knowledge base using Pinecone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant documents",
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Namespace of the Pinecone to search",
                                "default": "global",
                            },
                            "rerank_model": {
                                "type": "string",
                                "description": "Rerank model to use",
                                "enum": ["cohere-rerank-3.5"],
                                "default": "cohere-rerank-3.5",
                            },
                            "inner_limit": {
                                "type": "integer",
                                "description": "Maximum number of results to rerank",
                                "default": 10,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            self._tools.append(search_kb_tool)
            print("Knowledge base search capability added as default tool")

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
                {{"fact": "Topic X has property Y",
                    "relevance": "Important for understanding Z"}}
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
        self,
        user_id: str,
        user_text: str,
        timezone: str = None,
        original_user_text: str = None,
    ) -> AsyncGenerator[str, None]:
        """Process text input and stream AI responses asynchronously.

        Args:
            user_id (str): Unique identifier for the user/conversation.
            user_text (str): Text input from user to process.
            original_user_text (str, optional): Original user message for storage. If provided,
                                           this will be stored instead of user_text. Defaults to None.

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

        # For storage purposes, use original text if provided
        message_to_store = (
            original_user_text if original_user_text is not None else user_text
        )

        # Save the conversation to the database and Zep memory (if configured)
        metadata = {
            "user_id": user_id,
            "message": message_to_store,
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


class HumanAgent:
    """Represents a human operator in the agent swarm."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        specialization: str,
        notification_handler: Callable = None,
        availability_status: Literal["available",
                                     "busy", "offline"] = "available",
    ):
        """Initialize a human agent.

        Args:
            agent_id (str): Unique identifier for this human agent
            name (str): Display name of the human agent
            specialization (str): Area of expertise description
            notification_handler (Callable, optional): Function to call when agent receives a handoff
            availability_status (str): Current availability of the human agent
        """
        self.agent_id = agent_id
        self.name = name
        self.specialization = specialization
        self.notification_handler = notification_handler
        self.availability_status = availability_status
        self.current_tickets = {}  # Tracks tickets assigned to this human

    async def set_inactive_if_idle(self, idle_threshold_minutes=30):
        """Automatically set agent to offline if they've been idle for too long."""
        last_active = getattr(self, "last_active_time", None)

        if not last_active:
            return

        idle_time = datetime.datetime.now(datetime.timezone.utc) - last_active
        if idle_time > datetime.timedelta(minutes=idle_threshold_minutes):
            self.availability_status = "offline"
            return True
        return False

    async def record_activity(self):
        """Record when this human agent was last active."""
        self.last_active_time = datetime.datetime.now(datetime.timezone.utc)

        async def receive_handoff(
            self, ticket_id: str, user_id: str, query: str, context: str
        ) -> bool:
            """Handle receiving a ticket from an AI agent or another human."""
            if self.availability_status != "available":
                return False

            # Add to current tickets
            self.current_tickets[ticket_id] = {
                "user_id": user_id,
                "query": query,
                "context": context,
                "status": "pending",
                "received_at": datetime.datetime.now(datetime.timezone.utc),
            }

            # Record this activity
            await self.record_activity()

            # Notify the human operator through the configured handler
            if self.notification_handler:
                await self.notification_handler(
                    agent_id=self.agent_id,
                    ticket_id=ticket_id,
                    user_id=user_id,
                    query=query,
                    context=context,
                )

            # Set a reminder notification after 15 minutes if not handled
            asyncio.create_task(self._set_reminder(ticket_id, 15))

            return True

    async def _set_reminder(self, ticket_id: str, minutes: int):
        """Set a reminder for an unhandled ticket."""
        await asyncio.sleep(minutes * 60)  # Convert to seconds

        # Check if ticket still exists and is pending
        if (
            ticket_id in self.current_tickets
            and self.current_tickets[ticket_id].get("status") == "pending"
        ):
            # Send reminder notification
            if self.notification_handler:
                await self.notification_handler(
                    agent_id=self.agent_id,
                    ticket_id=ticket_id,
                    reminder=True,
                    minutes=minutes,
                )

    async def respond(self, ticket_id: str, response: str) -> Dict[str, Any]:
        """Submit a response to a user query.

        Args:
            ticket_id: The ticket identifier
            response: The human agent's response text

        Returns:
            Dict with response details and status
        """
        if ticket_id not in self.current_tickets:
            return {"status": "error", "message": "Ticket not found"}

        ticket = self.current_tickets[ticket_id]
        ticket["response"] = response
        ticket["response_time"] = datetime.datetime.now(datetime.timezone.utc)
        ticket["status"] = "responded"

        return {
            "status": "success",
            "ticket_id": ticket_id,
            "user_id": ticket["user_id"],
            "response": response,
        }

    async def handoff_to(
        self, ticket_id: str, target_agent_id: str, reason: str
    ) -> bool:
        """Hand off a ticket to another agent (AI or human).

        Args:
            ticket_id: The ticket to hand off
            target_agent_id: Agent to transfer the ticket to
            reason: Reason for the handoff

        Returns:
            bool: Whether handoff was successful
        """
        if ticket_id not in self.current_tickets:
            return False

        # This just marks it for handoff - the actual handoff is handled by the swarm
        self.current_tickets[ticket_id]["status"] = "handoff_requested"
        self.current_tickets[ticket_id]["handoff_target"] = target_agent_id
        self.current_tickets[ticket_id]["handoff_reason"] = reason

        return True

    def update_availability(
        self, status: Literal["available", "busy", "offline"]
    ) -> None:
        """Update the availability status of this human agent."""
        self.availability_status = status

    async def check_pending_tickets(self) -> str:
        """Get a summary of pending tickets for this human agent."""
        if not self.current_tickets:
            return "You have no pending tickets."

        pending_count = sum(
            1 for t in self.current_tickets.values() if t["status"] == "pending"
        )
        result = [f"You have {pending_count} pending tickets"]

        # Add details for each pending ticket
        for ticket_id, ticket in self.current_tickets.items():
            if ticket["status"] == "pending":
                received_time = ticket.get(
                    "received_at", datetime.datetime.now(datetime.timezone.utc)
                )
                time_diff = datetime.datetime.now(
                    datetime.timezone.utc) - received_time
                hours_ago = round(time_diff.total_seconds() / 3600, 1)

                result.append(
                    f"- Ticket {ticket_id[:8]}... from user {ticket['user_id'][:8]}... ({hours_ago}h ago)"
                )
                result.append(
                    f"  Query: {ticket['query'][:50]}..."
                    if len(ticket["query"]) > 50
                    else f"  Query: {ticket['query']}"
                )

        if pending_count == 0:
            result.append("No pending tickets requiring your attention.")

        return "\n".join(result)


class Swarm:
    """An AI Agent Swarm that coordinates specialized AI agents with handoff capabilities."""

    def __init__(
        self,
        database: MongoDatabase,
        directive: str = None,
        router_model: str = "gpt-4o-mini",
        insight_model: str = "gpt-4o-mini",
        enable_collective_memory: bool = True,
        enable_critic: bool = True,
        default_timezone: str = "UTC",
    ):
        """Initialize the multi-agent system with a shared database.

        Args:
            database (MongoDatabase): Shared MongoDB database instance
            directive (str, optional): Core directive/mission that governs all agents. Defaults to None.
            router_model (str, optional): Model to use for routing decisions. Defaults to "gpt-4o-mini".
            insight_model (str, optional): Model to extract collective insights. Defaults to "gpt-4o-mini".
            enable_collective_memory (bool, optional): Whether to enable collective memory. Defaults to True.
            enable_critic (bool, optional): Whether to enable the critic system. Defaults to True.
            default_timezone (str, optional): Default timezone for time-awareness. Defaults to "UTC".
        """
        self.agents = {}  # name -> AI instance
        self.specializations = {}  # name -> description
        self.database = database
        self.router_model = router_model
        self.insight_model = insight_model
        self.enable_collective_memory = enable_collective_memory
        self.default_timezone = default_timezone
        self.enable_critic = enable_critic

        # Initialize background tasks
        self._background_tasks = []
        self._shutdown_event = asyncio.Event()

        # Start ticket monitoring task
        self._start_background_tasks()

        # Store swarm directive
        self.swarm_directive = (
            directive
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

        # Create NPS survey collection
        if "nps_surveys" not in self.database.db.list_collection_names():
            self.database.db.create_collection("nps_surveys")
        self.nps_surveys = self.database.db["nps_surveys"]

        try:
            # Create index for NPS analytics
            self.nps_surveys.create_index([("agent_name", 1)])
            self.nps_surveys.create_index([("score", 1)])
            self.nps_surveys.create_index([("timestamp", 1)])
            print("Created indexes for NPS analytics")
        except Exception as e:
            print(f"Warning: NPS index creation might have failed: {e}")

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

    def register_human_agent(
        self,
        agent_id: str,
        name: str,
        specialization: str,
        notification_handler: Callable = None,
    ) -> HumanAgent:
        """Register a human agent with the swarm.

        Args:
            agent_id: Unique identifier for this human agent
            name: Display name of the human agent
            specialization: Description of expertise
            notification_handler: Function to call when agent receives handoff

        Returns:
            The created HumanAgent instance
        """
        # Create human agent instance
        human_agent = HumanAgent(
            agent_id=agent_id,
            name=name,
            specialization=specialization,
            notification_handler=notification_handler,
        )

        # Store in humans registry
        if not hasattr(self, "human_agents"):
            self.human_agents = {}
        self.human_agents[agent_id] = human_agent

        # Add human agent to specialization map
        self.specializations[agent_id] = f"[HUMAN] {specialization}"

        # Create or update the ticket collection
        if "tickets" not in self.database.db.list_collection_names():
            self.database.db.create_collection("tickets")
        self.tickets = self.database.db["tickets"]

        print(
            f"Registered human agent: {name}, specialization: {specialization[:50]}..."
        )

        # Update AI agents with human handoff capabilities
        self._update_all_handoff_capabilities()

        return human_agent

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Create a task to periodically check ticket timeouts
        task = asyncio.create_task(self._run_ticket_timeout_checker())
        self._background_tasks.append(task)

    async def _run_ticket_timeout_checker(self):
        """Background task to check for ticket timeouts."""
        while not self._shutdown_event.is_set():
            try:
                await self.check_ticket_timeouts()
            except Exception as e:
                print(f"Error checking ticket timeouts: {e}")

            # Check every 5 minutes
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                pass  # Continue the loop

    # Add shutdown method for clean task termination
    async def shutdown(self):
        """Gracefully shut down the swarm and its background tasks."""
        self._shutdown_event.set()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        async def _send_nps_survey(
            self, user_id: str, ticket_id: str, agent_name: str
        ) -> str:
            """Send NPS survey to user when a ticket is resolved."""
            # Create a simple numeric survey ID instead of complex UUID
            # Just use a short 6-digit ID
            survey_id = str(uuid.uuid4().int)[:6]

            # Store pending survey in database
            self.nps_surveys.insert_one(
                {
                    "survey_id": survey_id,
                    "user_id": user_id,
                    "ticket_id": ticket_id,
                    "agent_name": agent_name,
                    "status": "pending",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                }
            )

            # Create simple survey message
            survey_message = (
                f"""How was your experience? Rate 0-10: !rate {survey_id} [0-10]"""
            )

            # Save as separate system message in the database with ALL fields required by the interface
            message_id = str(uuid.uuid4())
            self.database.save_message(
                user_id,
                {
                    "id": message_id,  # Add explicit ID field
                    "user_id": user_id,
                    "message": "rate_request",
                    "response": survey_message,
                    "agent_name": agent_name,
                    "survey_id": survey_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "is_system_message": True,
                    "rating_submitted": False,  # Explicitly set to false initially
                },
            )

            # Return empty string since we don't need to append to agent response
            return ""

    def get_nps_metrics(
        self,
        agent_name: str = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ) -> dict:
        """Calculate NPS metrics overall or for a specific agent.

        Args:
            agent_name: Optional agent name to filter by
            start_date: Optional start date for date range
            end_date: Optional end date for date range

        Returns:
            Dictionary with NPS metrics
        """
        # Build query
        query = {"status": "completed"}

        if agent_name:
            query["agent_name"] = agent_name

        if start_date or end_date:
            query["completed_at"] = {}
            if start_date:
                query["completed_at"]["$gte"] = start_date
            if end_date:
                query["completed_at"]["$lte"] = end_date

        # Get all responses matching criteria
        responses = list(self.nps_surveys.find(query))

        if not responses:
            return {
                "nps_score": 0,
                "promoters": 0,
                "passives": 0,
                "detractors": 0,
                "total_responses": 0,
                "avg_score": 0,
            }

        # Count each category
        promoters = sum(1 for r in responses if r.get("score", 0) >= 9)
        passives = sum(1 for r in responses if 7 <= r.get("score", 0) <= 8)
        detractors = sum(1 for r in responses if r.get("score", 0) <= 6)

        total = len(responses)

        # Calculate NPS (percentage of promoters minus percentage of detractors)
        nps_score = int(((promoters - detractors) / total) * 100)

        # Calculate average score
        avg_score = sum(r.get("score", 0) for r in responses) / total

        # If agent_name was not specified, also get per-agent breakdown
        agent_breakdown = None
        if not agent_name:
            agent_breakdown = {}
            # Group by agent name
            pipeline = [
                {"$match": {"status": "completed"}},
                {
                    "$group": {
                        "_id": "$agent_name",
                        "avg_score": {"$avg": "$score"},
                        "count": {"$sum": 1},
                    }
                },
            ]
            for result in self.nps_surveys.aggregate(pipeline):
                agent_breakdown[result["_id"]] = {
                    "avg_score": round(result["avg_score"], 2),
                    "count": result["count"],
                }

        return {
            "nps_score": nps_score,
            "promoters": promoters,
            "promoters_pct": round((promoters / total) * 100, 1),
            "passives": passives,
            "passives_pct": round((passives / total) * 100, 1),
            "detractors": detractors,
            "detractors_pct": round((detractors / total) * 100, 1),
            "total_responses": total,
            "avg_score": round(avg_score, 2),
            "agent_breakdown": agent_breakdown,
        }

    async def _process_nps_command(self, user_id: str, command: str) -> str:
        """Process NPS survey responses with simplified format."""
        parts = command.strip().split(" ")

        # Handle simplified !rate command
        if command.startswith("!rate ") and len(parts) >= 3:
            survey_id = parts[1]

            # Find the survey
            survey = self.nps_surveys.find_one(
                {"survey_id": survey_id, "status": "pending"}
            )
            if not survey:
                return "⚠️ Invalid or expired rating ID."

            try:
                score = int(parts[2])
                if not 0 <= score <= 10:
                    raise ValueError()
            except ValueError:
                return "⚠️ Please provide a valid rating between 0-10."

            # Get feedback if provided
            feedback = " ".join(parts[3:]) if len(parts) > 3 else ""

            # Update survey with response
            self.nps_surveys.update_one(
                {"survey_id": survey_id},
                {
                    "$set": {
                        "score": score,
                        "feedback": feedback,
                        "status": "completed",
                        "completed_at": datetime.datetime.now(datetime.timezone.utc),
                    }
                },
            )

            return "✅ Thank you for your feedback! Your rating has been recorded."

    def _update_all_handoff_capabilities(self):
        """Update all agents with current handoff capabilities for both AI and human agents."""
        # Get all AI agent names
        ai_agent_names = list(self.agents.keys())

        # Get all human agent names
        human_agent_names = (
            list(self.human_agents.keys()) if hasattr(
                self, "human_agents") else []
        )

        # For each AI agent, update its handoff tools
        for agent_name, agent in self.agents.items():
            # Get available target agents (both AI and human)
            available_ai_targets = [
                name for name in ai_agent_names if name != agent_name
            ]
            available_targets = available_ai_targets + human_agent_names

            # First remove any existing handoff tools
            agent._tools = [
                t for t in agent._tools if t["function"]["name"] != "request_handoff"
            ]

            # Create updated handoff tool with both AI and human targets
            def create_handoff_tool(current_agent_name, available_targets_list):
                def request_handoff(target_agent: str, reason: str) -> str:
                    """Request an immediate handoff to another agent (AI or human).
                    This is an INTERNAL SYSTEM TOOL. The user will NOT see your reasoning about the handoff.
                    Use this tool IMMEDIATELY when a query is outside your expertise.

                    Args:
                        target_agent: Name of agent to transfer to. MUST be one of: {', '.join(available_targets_list)}.
                          DO NOT INVENT NEW NAMES OR VARIATIONS. Use EXACTLY one of these names.
                        reason: Brief explanation of why this question requires the specialist

                    Returns:
                        str: Empty string - the handoff is handled internally
                    """
                    # Validate target agent exists (either AI or human)
                    is_human_target = target_agent in human_agent_names
                    is_ai_target = target_agent in ai_agent_names

                    if not (is_human_target or is_ai_target):
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

                    # Set handoff info - now includes flag for whether target is human
                    agent._handoff_info = {
                        "target": target_agent,
                        "reason": reason,
                        "is_human_target": is_human_target,
                    }

                    # Return empty string - the actual handoff happens in the process method
                    return ""

                return request_handoff

            # Use the factory to create a properly-bound tool function
            handoff_tool = create_handoff_tool(agent_name, available_targets)

            # Initialize handoff info attribute
            agent._handoff_info = None

            # Add the updated handoff tool with proper closure
            agent.add_tool(handoff_tool)

            # Update agent instructions with handoff guidance including human agents
            ai_handoff_examples = "\n".join(
                [
                    f"  - `{name}` (AI: {self.specializations[name][:40]}...)"
                    for name in available_ai_targets
                ]
            )
            human_handoff_examples = "\n".join(
                [
                    f"  - `{name}` (HUMAN: {self.specializations[name].replace('[HUMAN] ', '')[:40]}...)"
                    for name in human_agent_names
                ]
            )

            handoff_instructions = f"""
            STRICT HANDOFF GUIDANCE:
            1. You must use ONLY the EXACT agent names listed below for handoffs.

            AI AGENTS (available immediately):
            {ai_handoff_examples}

            HUMAN AGENTS (might have response delay):
            {human_handoff_examples}

            2. DO NOT INVENT OR MODIFY AGENT NAMES.

            3. ONLY these EXACT agent names will work for handoffs: {', '.join(available_targets)}

            4. Use human agents ONLY when:
               - The question truly requires human judgment or expertise
               - The user explicitly asks for a human agent
               - The task involves confidential information that AI shouldn't access
            """

            # Update agent instructions with handoff guidance
            agent._instructions = (
                re.sub(
                    r"STRICT HANDOFF GUIDANCE:.*?(?=\n\n)",
                    handoff_instructions,
                    agent._instructions,
                    flags=re.DOTALL,
                )
                if "STRICT HANDOFF GUIDANCE" in agent._instructions
                else agent._instructions + "\n\n" + handoff_instructions
            )

        print("Updated handoff capabilities for all agents with AI and human targets")

    async def process_human_response(
        self,
        human_agent_id: str,
        ticket_id: str,
        response: str,
        handoff_to: str = None,
        handoff_reason: str = None,
    ) -> Dict[str, Any]:
        """Process a response from a human agent.

        Args:
            human_agent_id: ID of the human agent responding
            ticket_id: Ticket identifier
            response: Human agent's response text
            handoff_to: Optional target agent for handoff
            handoff_reason: Optional reason for handoff

        Returns:
            Dict with status and details
        """
        # Verify the human agent exists
        if not hasattr(self, "human_agents") or human_agent_id not in self.human_agents:
            return {"status": "error", "message": "Human agent not found"}

        human_agent = self.human_agents[human_agent_id]

        # Record human agent activity
        await human_agent.record_activity()

        # Get the ticket
        ticket = self.tickets.find_one({"_id": ticket_id})
        if not ticket:
            return {"status": "error", "message": "Ticket not found"}

        # Check if ticket is assigned to this agent
        if ticket.get("assigned_to") != human_agent_id:
            return {"status": "error", "message": "Ticket not assigned to this agent"}

        # If handoff requested
        if handoff_to:
            # Determine if target is human or AI
            is_human_target = (
                hasattr(self, "human_agents") and handoff_to in self.human_agents
            )

            is_ai_target = handoff_to in self.agents

            if not (is_human_target or is_ai_target):
                return {"status": "error", "message": "Invalid handoff target"}

            # Record the handoff
            self.handoffs.insert_one(
                {
                    "ticket_id": ticket_id,
                    "user_id": ticket["user_id"],
                    "from_agent": human_agent_id,
                    "to_agent": handoff_to,
                    "reason": handoff_reason or "Human agent handoff",
                    "query": ticket["query"],
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                }
            )

            # Update ticket status
            self.tickets.update_one(
                {"_id": ticket_id},
                {
                    "$set": {
                        "assigned_to": handoff_to,
                        "status": "transferred",
                        "human_response": response,
                        "handoff_reason": handoff_reason,
                        "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    }
                },
            )

            # Process based on target type
            if is_human_target:
                # Human-to-human handoff
                target_human = self.human_agents[handoff_to]

                # Get updated context including the human's response
                context = (
                    ticket.get("context", "")
                    + f"\n\nHuman agent {human_agent.name}: {response}"
                )

                # Try to hand off to the human agent
                accepted = await target_human.receive_handoff(
                    ticket_id=ticket_id,
                    user_id=ticket["user_id"],
                    query=ticket["query"],
                    context=context,
                )

                if accepted:
                    return {
                        "status": "success",
                        "message": f"Transferred to human agent {target_human.name}",
                        "ticket_id": ticket_id,
                    }
                else:
                    return {
                        "status": "warning",
                        "message": f"Human agent {target_human.name} is unavailable",
                    }
            else:
                # Human-to-AI handoff
                target_ai = self.agents[handoff_to]

                # Return details for AI processing
                return {
                    "status": "success",
                    "message": f"Transferred to AI agent {handoff_to}",
                    "ticket_id": ticket_id,
                    "ai_agent": target_ai,
                    "user_id": ticket["user_id"],
                    "query": ticket["query"],
                }

        # No handoff - just record the human response
        self.tickets.update_one(
            {"_id": ticket_id},
            {
                "$set": {
                    "status": "resolved",
                    "human_response": response,
                    "resolved_at": datetime.datetime.now(datetime.timezone.utc),
                }
            },
        )

        # Also record in messages for continuity
        self.database.save_message(
            ticket["user_id"],
            {
                "message": ticket["query"],
                "response": response,
                "human_agent": human_agent_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
            },
        )

        return {
            "status": "success",
            "message": "Response recorded",
            "ticket_id": ticket_id,
        }

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
        """

        # Extract insights using AI with structured parsing
        try:
            # Parse the response using the Pydantic model
            completion = first_agent._client.beta.chat.completions.parse(
                model=self.insight_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract important factual insights from conversations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=CollectiveMemoryResponse,
                temperature=0.1,
            )

            # Extract the Pydantic model
            memory_response = completion.choices[0].message.parsed

            # Store in MongoDB (keeps all metadata and text)
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            mongo_records = []

            for insight in memory_response.insights:
                record_id = str(uuid.uuid4())
                record = {
                    "_id": record_id,
                    "fact": insight.fact,
                    "relevance": insight.relevance,
                    "timestamp": timestamp,
                    "source_user_id": user_id,
                }
                mongo_records.append(record)

            if mongo_records:
                for record in mongo_records:
                    self.collective_memory.insert_one(record)

            # Also store in Pinecone for semantic search if available
            if (
                mongo_records
                and hasattr(first_agent, "_pinecone")
                and first_agent._pinecone
                and first_agent.kb
            ):
                try:
                    # Generate embeddings
                    texts = [
                        f"{record['fact']}: {record['relevance']}"
                        for record in mongo_records
                    ]
                    embeddings = first_agent._pinecone.inference.embed(
                        model=first_agent._pinecone_embedding_model,
                        inputs=texts,
                        parameters={"input_type": "passage",
                                    "truncate": "END"},
                    )

                    # Create vectors for Pinecone
                    vectors = []
                    for record, embedding in zip(mongo_records, embeddings):
                        vectors.append(
                            {
                                "id": record["_id"],
                                "values": embedding.values,
                                "metadata": {
                                    "fact": record["fact"],
                                    "relevance": record["relevance"],
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
                        f"Stored {len(mongo_records)} insights in both MongoDB and Pinecone"
                    )
                except Exception as e:
                    print(f"Error storing insights in Pinecone: {e}")
            else:
                print(f"Stored {len(mongo_records)} insights in MongoDB only")

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
                            model=first_agent._pinecone_embedding_model,
                            inputs=[query],
                            parameters={"input_type": "passage",
                                        "truncate": "END"},
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

    async def process_human_message(
        self, human_agent_id: str, message: str, target_agent: str = None
    ) -> AsyncGenerator[str, None]:
        """Process a message initiated by a human agent without creating a ticket.

        This is used for human agent questions to the AI, not for user support tickets.
        """
        # Verify the human agent exists
        if not hasattr(self, "human_agents") or human_agent_id not in self.human_agents:
            yield "Error: Human agent not found."
            return

        human_agent = self.human_agents[human_agent_id]

        # Create a special prefix to mark this as a human agent message
        prefixed_message = (
            f"[INTERNAL TEAM MESSAGE FROM HUMAN AGENT {human_agent.name}]: {message}"
        )

        # Determine which agent should handle this
        if target_agent and target_agent in self.agents:
            ai_agent = self.agents[target_agent]
        else:
            # Use the router to find the best agent
            first_agent = next(iter(self.agents.values()))
            best_agent_name = await self._get_routing_decision(first_agent, message)
            ai_agent = self.agents[best_agent_name]

        # Generate response from the AI agent
        async for chunk in ai_agent.text(human_agent_id, prefixed_message):
            yield chunk

        # Store in a separate collection with is_human_agent_message flag
        self.database.save_message(
            human_agent_id,
            {
                "id": str(uuid.uuid4()),
                "user_id": human_agent_id,
                "message": message,
                "is_human_agent_message": True,  # Flag to distinguish from regular messages
                "target_agent": target_agent or ai_agent.__class__.__name__,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
            },
        )

    async def _process_human_agent_commands(
        self, user_id: str, command: str
    ) -> AsyncGenerator[str, None]:
        """Process commands from human agents."""

        # Check if user is a registered human agent
        is_human_agent = False
        if hasattr(self, "human_agents"):
            for agent_id, agent in self.human_agents.items():
                if agent_id == user_id:
                    is_human_agent = True
                    break

        if not is_human_agent:
            yield "⚠️ Only registered human agents can use these commands."
            return

        # Handle !ask command
        if command.startswith("!ask "):
            parts = command.strip().split(" ", 2)
            if len(parts) < 2:
                yield "⚠️ Format: !ask [agent_name or 'any'] your question here"
                return

            target = None
            message = parts[1]

            # Check if a specific agent is targeted
            if len(parts) > 2:
                potential_target = parts[1]
                if (
                    potential_target.lower() != "any"
                    and potential_target in self.agents
                ):
                    target = potential_target
                    message = parts[2]
                else:
                    message = " ".join(parts[1:])

            # Use process_human_message which doesn't create tickets
            async for chunk in self.process_human_message(user_id, message, target):
                yield chunk
            return

    async def check_ticket_timeouts(self):
        """Periodically check for and handle stalled human agent tickets."""
        # Find tickets assigned to human agents that have been pending too long
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=1
        )

        stalled_tickets = self.tickets.find(
            {
                "assigned_to": {"$in": list(self.human_agents.keys())},
                "status": "pending",
                "updated_at": {"$lt": cutoff_time},
            }
        )

        for ticket in stalled_tickets:
            human_agent_id = ticket["assigned_to"]
            human_agent = self.human_agents.get(human_agent_id)

            if not human_agent or human_agent.availability_status != "available":
                # Find a suitable AI agent to reassign to
                first_agent = next(iter(self.agents.values()))
                agent_name = await self._get_routing_decision(
                    first_agent, ticket["query"]
                )

                # Update ticket with selected agent
                self.tickets.update_one(
                    {"_id": ticket["_id"]},
                    {
                        "$set": {
                            "assigned_to": agent_name,
                            "status": "transferred",
                            "handoff_reason": "Auto-reassigned due to human agent unavailability",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )

                # Log the auto-reassignment
                self.handoffs.insert_one(
                    {
                        "ticket_id": ticket["_id"],
                        "user_id": ticket["user_id"],
                        "from_agent": human_agent_id,
                        "to_agent": agent_name,
                        "reason": "Auto-reassigned due to human agent unavailability",
                        "query": ticket["query"],
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                        "automatic": True,
                    }
                )

                print(
                    f"Auto-reassigned ticket {ticket['_id']} from {human_agent_id} to {agent_name}"
                )

    async def _process_agent_directory_command(self, user_id: str) -> str:
        """Provide a directory of available AI agents for human agents."""
        # Check if user is a registered human agent
        is_human_agent = False
        if hasattr(self, "human_agents") and user_id in self.human_agents:
            is_human_agent = True

        if not is_human_agent:
            return "⚠️ Only registered human agents can access the agent directory."

        # Format AI agents directory
        directory = ["## 🤖 AI Agent Directory", ""]
        directory.append("### Available AI Agents")
        directory.append("")

        # List all AI agents with their specializations
        for agent_name, specialization in self.specializations.items():
            # Skip human agents in this section
            if hasattr(self, "human_agents") and agent_name in self.human_agents:
                continue

            directory.append(f"**@{agent_name}** - {specialization}")

        # Add section for human agents if any exist
        if hasattr(self, "human_agents") and self.human_agents:
            directory.append("")
            directory.append("### 👤 Human Agents")
            directory.append("")

            for agent_id, agent in self.human_agents.items():
                status_emoji = {"available": "🟢", "busy": "🟠", "offline": "⚫"}.get(
                    agent.availability_status, "⚫"
                )

                directory.append(
                    f"**@{agent_id}** - {status_emoji} {agent.name}: {agent.specialization}"
                )

        # Add usage instructions
        directory.append("")
        directory.append("### How to Use")
        directory.append("")
        directory.append("**Direct messaging to AI agents:**")
        directory.append(
            "- Type `@agent_name your message` to send a direct message to a specific agent"
        )
        directory.append(
            "- Example: `@developer How do I implement a Solana program?`")
        directory.append("")
        directory.append("**Human agents can also:**")
        directory.append(
            "- Use the ticket system: `!ticket list`, `!ticket view`, etc."
        )

        return "\n".join(directory)

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle ticket management.

        Args:
            user_id (str): Unique user identifier
            user_text (str): User's text input
            timezone (str, optional): User-specific timezone
        """
        try:
            # First, check if sender is a human agent
            is_human_agent = False
            if hasattr(self, "human_agents") and user_id in self.human_agents:
                is_human_agent = True

            # Handle human agent messages differently (no ticket creation)
            if is_human_agent:
                # Handle specific human agent commands
                if user_text.lower() == "!agents":
                    yield await self._process_agent_directory_command(user_id)
                    return

                # Parse for target agent specification if available
                target_agent = None
                message = user_text

                # Check if message starts with @agent_name to target specific agent
                if user_text.startswith("@"):
                    parts = user_text.split(" ", 1)
                    potential_target = parts[0][1:]  # Remove the @ symbol
                    if potential_target in self.agents:
                        target_agent = potential_target
                        message = parts[1] if len(parts) > 1 else ""

                # Process as human agent message (no ticket creation)
                async for chunk in self.process_human_message(
                    user_id, message, target_agent
                ):
                    yield chunk
                return

            # Standard user commands handling (for regular users)
            # Handle NPS survey responses
            if user_text.lower().startswith("!rate "):
                yield await self._process_nps_command(user_id, user_text)
                return

            # Handle special ticket management commands
            if user_text.lower().startswith("!ticket"):
                yield await self._process_ticket_commands(user_id, user_text)
                return

            # Handle special collective memory commands
            if user_text.strip().lower().startswith("!memory "):
                query = user_text[8:].strip()
                yield self.search_collective_memory(query)
                return

            # Check for registered agents
            if not self.agents:
                yield "Error: No agents are registered with the system. Please register at least one agent first."
                return

            # Ensure tickets collection exists
            if "tickets" not in self.database.db.list_collection_names():
                self.database.db.create_collection("tickets")
            self.tickets = self.database.db["tickets"]

            # Check if this is continuing an existing ticket
            active_ticket = self.tickets.find_one(
                {
                    "user_id": user_id,
                    "status": {"$in": ["pending", "active", "transferred"]},
                }
            )

            ticket_id = None
            current_agent = None

            if active_ticket:
                # Continue with existing ticket
                ticket_id = active_ticket["_id"]
                current_agent_name = active_ticket.get("assigned_to")

                # Update ticket to active status if it was pending/transferred
                if active_ticket["status"] in ["pending", "transferred"]:
                    self.tickets.update_one(
                        {"_id": ticket_id},
                        {
                            "$set": {
                                "status": "active",
                                "last_activity": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                            }
                        },
                    )

                # If it was transferred to a specific agent, use that agent
                if current_agent_name and current_agent_name in self.agents:
                    current_agent = self.agents[current_agent_name]
                    print(
                        f"Continuing ticket {ticket_id} with agent {current_agent_name}"
                    )
                else:
                    # Get routing decision if no specific agent is assigned
                    first_agent = next(iter(self.agents.values()))
                    agent_name = await self._get_routing_decision(
                        first_agent, user_text
                    )
                    current_agent = self.agents[agent_name]

                    # Update ticket with selected agent
                    self.tickets.update_one(
                        {"_id": ticket_id},
                        {
                            "$set": {
                                "assigned_to": agent_name,
                                "updated_at": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                            }
                        },
                    )
                    print(f"Reassigned ticket {ticket_id} to {agent_name}")
            else:
                # Create new ticket for this interaction
                ticket_id = str(uuid.uuid4())

                # Get initial routing and agent
                first_agent = next(iter(self.agents.values()))
                agent_name = await self._get_routing_decision(first_agent, user_text)
                current_agent = self.agents[agent_name]

                # Get conversation context
                context = ""
                if hasattr(current_agent, "get_memory_context"):
                    context = current_agent.get_memory_context(user_id)

                # Store new ticket in database
                self.tickets.insert_one(
                    {
                        "_id": ticket_id,
                        "user_id": user_id,
                        "query": user_text,
                        "created_at": datetime.datetime.now(datetime.timezone.utc),
                        "assigned_to": agent_name,
                        "status": "active",
                        "context": context,
                    }
                )
                print(
                    f"Created new ticket {ticket_id}, assigned to {agent_name}")

            # Reset handoff info
            current_agent._handoff_info = None

            # Response tracking
            final_response = ""

            # Process response stream with ticket context
            async for chunk in self._stream_response(
                user_id, user_text, current_agent, timezone, ticket_id
            ):
                yield chunk
                final_response += chunk

            # Skip ticket resolution check if a handoff occurred during response
            if not current_agent._handoff_info:
                # Check if ticket should be resolved based on AI's response
                resolution = await self._check_ticket_resolution(
                    user_id, final_response, ticket_id
                )

                # Update ticket status based on resolution
                if resolution.status == "resolved" and resolution.confidence >= 0.7:
                    self.tickets.update_one(
                        {"_id": ticket_id},
                        {
                            "$set": {
                                "status": "resolved",
                                "resolution_confidence": resolution.confidence,
                                "resolution_reasoning": resolution.reasoning,
                                "resolved_at": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                            }
                        },
                    )
                    print(
                        f"Ticket {ticket_id} marked as resolved with confidence {resolution.confidence}"
                    )
                    # Send NPS survey after resolution
                    await self._send_nps_survey(user_id, ticket_id, agent_name)
                else:
                    # Update with pending status
                    self.tickets.update_one(
                        {"_id": ticket_id},
                        {
                            "$set": {
                                "status": "pending_confirmation",
                                "resolution_confidence": resolution.confidence,
                                "resolution_reasoning": resolution.reasoning,
                                "suggested_actions": resolution.suggested_actions,
                                "updated_at": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                            }
                        },
                    )
                    print(
                        f"Ticket {ticket_id} needs followup (confidence: {resolution.confidence})"
                    )

            # Post-processing: learn from conversation
            conversation = {
                "user_id": user_id,
                "message": user_text,
                "response": final_response,
                "ticket_id": ticket_id,
            }

            # Run post-processing tasks concurrently
            tasks = []

            # Add collective memory task if enabled
            if self.enable_collective_memory:
                tasks.append(self.extract_and_store_insights(
                    user_id, conversation))

            # Run all post-processing tasks concurrently without waiting
            if tasks:
                asyncio.create_task(self._run_post_processing_tasks(tasks))

        except Exception as e:
            print(f"Error in multi-agent processing: {str(e)}")
            print(traceback.format_exc())
            yield "\n\nI apologize for the technical difficulty.\n\n"

    async def _check_ticket_resolution(self, user_id, response, ticket_id):
        """Determine if a ticket can be resolved based on the AI response using structured output.

        Args:
            user_id: The user identifier
            response: The AI agent's response to evaluate
            ticket_id: The ticket identifier

        Returns:
            TicketResolution: Structured resolution data with status, confidence and reasoning
        """
        # Get first agent to use its client
        first_agent = next(iter(self.agents.values()))

        # Get ticket details
        ticket = self.tickets.find_one({"_id": ticket_id})
        if not ticket:
            return TicketResolution(
                status="cannot_determine",
                confidence=0.0,
                reasoning="Ticket not found in database",
            )

        prompt = f"""
        You are evaluating if a user's question has been fully addressed.
        
        Original query: {ticket.get('query', 'Unknown')}
        
        Agent response: {response}
        
        Analyze how well the response addresses the query and provide a structured assessment.
        """

        try:
            # Use structured parsing to get detailed resolution information
            resolution_response = first_agent._client.beta.chat.completions.parse(
                model=self.router_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Provide a structured assessment of this response.",
                    },
                ],
                response_format=TicketResolution,
                temperature=0.1,
            )

            return resolution_response.choices[0].message.parsed

        except Exception as e:
            print(f"Error checking ticket resolution: {e}")
            return TicketResolution(
                status="cannot_determine",
                confidence=0.0,
                reasoning=f"Error processing resolution check: {str(e)}",
            )

    async def _process_ticket_commands(self, user_id: str, command: str) -> str:
        """Process ticket management commands directly in chat."""
        parts = command.strip().split(" ", 2)

        # Check if user is a registered human agent
        is_human_agent = False
        human_agent = None
        if hasattr(self, "human_agents"):
            for agent_id, agent in self.human_agents.items():
                if agent_id == user_id:
                    is_human_agent = True
                    human_agent = agent
                    break

        if not is_human_agent:
            return "⚠️ Only registered human agents can use ticket commands."

        # Process various ticket commands
        if len(parts) > 1:
            action = parts[1].lower()

            # List tickets assigned to this human agent
            if action == "list":
                tickets = list(
                    self.tickets.find(
                        {"assigned_to": user_id, "status": "pending"})
                )

                if not tickets:
                    return "📋 You have no pending tickets."

                ticket_list = ["## Your Pending Tickets", ""]
                for i, ticket in enumerate(tickets, 1):
                    created = ticket.get(
                        "created_at", datetime.datetime.now(
                            datetime.timezone.utc)
                    )
                    time_ago = self._format_time_ago(created)

                    ticket_list.append(
                        f"**{i}. Ticket {ticket['_id'][:8]}...** ({time_ago})"
                    )
                    ticket_list.append(
                        f"Query: {ticket.get('query', 'No query')[:100]}..."
                    )
                    ticket_list.append("")

                return "\n".join(ticket_list)

            # View a specific ticket
            elif action == "view" and len(parts) > 2:
                ticket_id = parts[2]
                ticket = self.tickets.find_one(
                    {"_id": {"$regex": f"^{ticket_id}.*"}, "assigned_to": user_id}
                )

                if not ticket:
                    return f"⚠️ No ticket found with ID starting with '{ticket_id}'"

                context = ticket.get("context", "No previous context")
                query = ticket.get("query", "No query")
                created = ticket.get(
                    "created_at", datetime.datetime.now(datetime.timezone.utc)
                )
                time_ago = self._format_time_ago(created)

                return f"""## Ticket Details ({ticket['_id']})
    
    Status: {ticket.get('status', 'pending')}
    Created: {time_ago}
    
    ### User Query
    {query}
    
    ### Conversation Context
    {context}
    """

            # Respond to a ticket
            elif action == "respond" and len(parts) > 2:
                # Format: !ticket respond ticket_id response text here
                response_parts = parts[2].split(" ", 1)
                if len(response_parts) < 2:
                    return "⚠️ Format: !ticket respond ticket_id your response text"

                ticket_id = response_parts[0]
                response_text = response_parts[1]

                # Find the ticket
                ticket = self.tickets.find_one(
                    {"_id": {"$regex": f"^{ticket_id}.*"}, "assigned_to": user_id}
                )
                if not ticket:
                    return f"⚠️ No ticket found with ID starting with '{ticket_id}'"

                # Process the response
                response_result = await human_agent.respond(
                    ticket["_id"], response_text
                )

                # Check if response was successful
                if response_result.get("status") != "success":
                    return f"⚠️ Failed to respond to ticket: {response_result.get('message', 'Unknown error')}"

                # Update ticket and save response
                self.tickets.update_one(
                    {"_id": ticket["_id"]},
                    {
                        "$set": {
                            "status": "resolved",
                            "human_response": response_text,
                            "resolved_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )

                # Also record in messages for continuity
                self.database.save_message(
                    ticket["user_id"],
                    {
                        "message": ticket["query"],
                        "response": response_text,
                        "human_agent": user_id,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    },
                )

                return f"✅ Response recorded for ticket {ticket_id}. The ticket has been marked as resolved."

            # Transfer a ticket
            elif action == "transfer" and len(parts) > 2:
                # Format: !ticket transfer ticket_id target_agent [reason]
                transfer_parts = parts[2].split(" ", 2)
                if len(transfer_parts) < 2:
                    return "⚠️ Format: !ticket transfer ticket_id target_agent [reason]"

                ticket_id = transfer_parts[0]
                target_agent = transfer_parts[1]
                reason = (
                    transfer_parts[2]
                    if len(transfer_parts) > 2
                    else "Human agent transfer"
                )

                # Find the ticket
                ticket = self.tickets.find_one(
                    {"_id": {"$regex": f"^{ticket_id}.*"}, "assigned_to": user_id}
                )
                if not ticket:
                    return f"⚠️ No ticket found with ID starting with '{ticket_id}'"

                # Handle transfer logic
                # Determine if target is human or AI
                is_human_target = (
                    hasattr(
                        self, "human_agents") and target_agent in self.human_agents
                )

                is_ai_target = target_agent in self.agents

                if not (is_human_target or is_ai_target):
                    return f"⚠️ Invalid transfer target '{target_agent}'. Must be a valid agent name."

                # Record the handoff
                self.handoffs.insert_one(
                    {
                        "ticket_id": ticket["_id"],
                        "user_id": ticket["user_id"],
                        "from_agent": user_id,
                        "to_agent": target_agent,
                        "reason": reason,
                        "query": ticket["query"],
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    }
                )

                # Update ticket status in database
                self.tickets.update_one(
                    {"_id": ticket["_id"]},
                    {
                        "$set": {
                            "assigned_to": target_agent,
                            "status": "transferred",
                            "handoff_reason": reason,
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )

                # Process based on target type
                if is_human_target:
                    # Human-to-human handoff
                    target_human = self.human_agents[target_agent]

                    # Get updated context including the current human's notes
                    context = (
                        ticket.get("context", "")
                        + f"\n\nHuman agent {human_agent.name}: Transferring with note: {reason}"
                    )

                    # Try to hand off to the human agent
                    accepted = await target_human.receive_handoff(
                        ticket_id=ticket["_id"],
                        user_id=ticket["user_id"],
                        query=ticket["query"],
                        context=context,
                    )

                    if accepted:
                        return (
                            f"✅ Ticket transferred to human agent {target_human.name}"
                        )
                    else:
                        return f"⚠️ Human agent {target_human.name} is unavailable. Ticket is still transferred but pending their acceptance."
                else:
                    # Human-to-AI handoff
                    return f"✅ Ticket transferred to AI agent {target_agent}. The AI will handle this in the user's next interaction."

        # Help command or invalid format
        help_text = """
    ## Ticket Commands
    
    - `!ticket list` - Show your pending tickets
    - `!ticket view [ticket_id]` - View details of a specific ticket
    - `!ticket respond [ticket_id] [response]` - Respond to a ticket
    - `!ticket transfer [ticket_id] [target_agent] [reason]` - Transfer ticket to another agent
        """
        return help_text.strip()

    def _format_time_ago(self, timestamp):
        """Format a timestamp as a human-readable time ago string."""
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = now - timestamp

        if diff.days > 0:
            return f"{diff.days} days ago"

        hours, remainder = divmod(diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours} hours ago"
        if minutes > 0:
            return f"{minutes} minutes ago"
        return "just now"

    async def _stream_response(
        self, user_id, user_text, current_agent, timezone=None, ticket_id=None
    ) -> AsyncGenerator[str, None]:
        """Stream response from an agent, handling potential handoffs to AI or human agents."""
        handoff_detected = False
        response_started = False
        full_response = ""
        agent_name = None  # For the agent's name

        # Get agent name for recording purposes
        for name, agent in self.agents.items():
            if agent == current_agent:
                agent_name = name
                break

        # Get recent feedback for this agent to improve the response
        recent_feedback = []

        if self.enable_critic and hasattr(self, "critic"):
            # Get recent feedback for this specific agent
            recent_feedback = self.critic.get_agent_feedback(
                agent_name, limit=3)
            print(
                f"Retrieved {len(recent_feedback)} feedback items for agent {agent_name}"
            )

        # Augment user text with feedback instructions if available
        augmented_instruction = user_text
        if recent_feedback:
            # Create targeted improvement instructions based on past feedback
            feedback_summary = ""
            for feedback in recent_feedback:
                area = feedback.get("improvement_area", "Unknown")
                recommendation = feedback.get(
                    "recommendation", "No specific recommendation"
                )
                feedback_summary += f"- {area}: {recommendation}\n"

            # Add as hidden instructions to the agent
            augmented_instruction = f"""
            {user_text}
            
            [SYSTEM NOTE: Apply these improvements from recent feedback:
            {feedback_summary}
            The user will not see these instructions.]
            """
            print("Added feedback-based improvement instructions to prompt")

        async for chunk in current_agent.text(
            user_id, augmented_instruction, timezone, user_text
        ):
            # Accumulate the full response for critic analysis
            full_response += chunk

            # Check for handoff after each chunk
            if current_agent._handoff_info and not handoff_detected:
                handoff_detected = True
                target_name = current_agent._handoff_info["target"]
                reason = current_agent._handoff_info["reason"]
                is_human_target = current_agent._handoff_info.get(
                    "is_human_target", False
                )

                # Record the handoff without waiting
                asyncio.create_task(
                    self._record_handoff(
                        user_id,
                        agent_name or "unknown_agent",
                        target_name,
                        reason,
                        user_text,
                    )
                )

                # Add separator if needed
                if response_started:
                    yield "\n\n---\n\n"

                # Handle differently based on target type (AI vs human)
                if is_human_target and hasattr(self, "human_agents"):
                    # Create a ticket in the database
                    ticket_id = str(uuid.uuid4())

                    # Get conversation history
                    context = ""
                    if hasattr(current_agent, "get_memory_context"):
                        context = current_agent.get_memory_context(user_id)

                    # Store ticket in database
                    self.tickets.insert_one(
                        {
                            "_id": ticket_id,
                            "user_id": user_id,
                            "query": user_text,
                            "context": context,
                            "ai_response_before_handoff": full_response,
                            "assigned_to": target_name,
                            "status": "pending",
                            "created_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    )

                    # Get the human agent
                    human_agent = self.human_agents.get(target_name)

                    if human_agent:
                        # Try to hand off to the human agent
                        accepted = await human_agent.receive_handoff(
                            ticket_id=ticket_id,
                            user_id=user_id,
                            query=user_text,
                            context=context,
                        )

                        if accepted:
                            human_availability = {
                                "available": "available now",
                                "busy": "busy but will respond soon",
                                "offline": "currently offline but will respond when back",
                            }.get(
                                human_agent.availability_status,
                                "will respond when available",
                            )

                            # Provide a friendly handoff message to the user
                            handoff_message = f"""
                            I've transferred your question to {human_agent.name}, who specializes in {human_agent.specialization}.
                            
                            A human specialist will provide a more tailored response. They are {human_availability}.
                            
                            Your ticket ID is: {ticket_id}
                            """
                            yield handoff_message.strip()
                        else:
                            # Human agent couldn't accept - fall back to an AI agent
                            yield "I tried to transfer your question to a human specialist, but they're unavailable at the moment. Let me help you instead.\n\n"

                            # Get the first AI agent
                            fallback_agent = next(iter(self.agents.values()))

                            # Stream from fallback AI agent
                            async for new_chunk in fallback_agent.text(
                                user_id, user_text
                            ):
                                yield new_chunk
                                # Force immediate delivery
                                await asyncio.sleep(0)
                    else:
                        yield "I tried to transfer your question to a human specialist, but there was an error. Let me help you instead.\n\n"

                        # Fallback to first AI agent
                        fallback_agent = next(iter(self.agents.values()))
                        async for new_chunk in fallback_agent.text(user_id, user_text):
                            yield new_chunk
                            await asyncio.sleep(0)
                else:
                    # Standard AI-to-AI handoff
                    target_agent = self.agents[target_name]

                    # Update the ticket if we have one
                    if ticket_id:
                        self.tickets.update_one(
                            {"_id": ticket_id},
                            {
                                "$set": {
                                    "assigned_to": target_name,
                                    "status": "transferred",
                                    "handoff_reason": reason,
                                    "updated_at": datetime.datetime.now(
                                        datetime.timezone.utc
                                    ),
                                }
                            },
                        )
                        print(
                            f"Updated ticket {ticket_id}, transferred to {target_name}"
                        )

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
        if self.enable_critic and hasattr(self, "critic") and agent_name:
            # Schedule async analysis without blocking
            asyncio.create_task(
                self.critic.analyze_interaction(
                    agent_name=agent_name,
                    user_query=user_text,
                    response=full_response,
                )
            )
            print(f"Scheduled critic analysis for {agent_name} response")

    async def _run_post_processing_tasks(self, tasks):
        """Run multiple post-processing tasks concurrently."""
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error in post-processing tasks: {e}")

    async def _get_routing_decision(self, agent, user_text):
        """Get routing decision with NPS performance weighting."""

        # First, get candidate agents based on specialization
        enhanced_prompt = f"""
        Analyze this user query and return the TOP 2 MOST APPROPRIATE specialists.
        
        User query: "{user_text}"
        
        Available specialists:
        {json.dumps(self.specializations, indent=2)}
        
        CRITICAL ROUTING INSTRUCTIONS:
        1. For compound questions with multiple aspects spanning different domains,
           choose specialists who should address the CONCEPTUAL or EDUCATIONAL aspects first.
        
        2. Choose implementation specialists only when the query is PURELY about implementation.
        
        3. Return EXACTLY two specialist names in order of relevance, comma-separated.
           Format: "specialist1, specialist2"
        """

        # Get top candidates based on specialization match
        router_response = agent._client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.2,
        )

        raw_response = router_response.choices[0].message.content.strip()
        candidates = [name.strip() for name in raw_response.split(",")][:2]
        validated_candidates = [
            self._match_agent_name(name) for name in candidates]

        # If only one candidate or all candidates are the same, return that
        if len(set(validated_candidates)) == 1:
            return validated_candidates[0]

        # Apply NPS weighting for final selection
        weights = {}
        for candidate in validated_candidates:
            # Get NPS metrics for this agent
            metrics = self.get_nps_metrics(candidate)

            # Base score from NPS - default to 0.5 if no data
            if (
                metrics["total_responses"] > 5
            ):  # Require minimum responses for reliability
                nps_score = metrics["avg_score"] / \
                    10.0  # Normalize to 0-1 range
            else:
                nps_score = 0.5  # Default for agents with insufficient data

            # Weight by position in candidate list (first candidate gets priority)
            position_weight = 0.7 if candidate == validated_candidates[0] else 0.3

            # Final weighting
            weights[candidate] = (position_weight * 0.7) + (nps_score * 0.3)

        # Select probabilistically based on weights
        agents = list(weights.keys())
        weight_values = list(weights.values())

        # Normalize weights
        total = sum(weight_values)
        if total > 0:
            normalized_weights = [w / total for w in weight_values]
            selected = random.choices(
                agents, weights=normalized_weights, k=1)[0]
        else:
            selected = validated_candidates[0]

        print(f"Selected {selected} with weights: {weights}")
        return selected

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
