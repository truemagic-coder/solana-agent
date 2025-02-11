import asyncio
from datetime import datetime
import json
from typing import AsyncGenerator, List, Literal, Optional, Dict, Any, Callable
import uuid
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from openai import AssistantEventHandler
from openai.types.beta.threads import TextDelta, Text
from typing_extensions import override
import inspect
import requests
from zep_cloud.client import AsyncZep
from zep_cloud.client import Zep
from zep_cloud.types import Message
import pandas as pd
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
        self._client = AsyncIOMotorClient(db_url)
        self._db = self._client[db_name]
        self._threads = self._db["threads"]
        self.messages = self._db["messages"]

    async def save_thread_id(self, user_id: str, thread_id: str):
        await self._threads.insert_one({"thread_id": thread_id, "user_id": user_id})

    async def get_thread_id(self, user_id: str) -> Optional[str]:
        document = await self._threads.find_one({"user_id": user_id})
        return document["thread_id"] if document else None

    async def save_message(self, user_id: str, metadata: Dict[str, Any]):
        metadata["user_id"] = user_id
        await self.messages.insert_one(metadata)

    async def delete_all_threads(self):
        await self._threads.delete_many({})

    async def clear_user_history(self, user_id: str):
        await self.messages.delete_many({"user_id": user_id})
        await self._threads.delete_one({"user_id": user_id})


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
        gemini_api_key: str = None,
        pinecone_api_key: str = None,
        pinecone_index_name: str = None,
        code_interpreter: bool = True,
        openai_assistant_model: Literal["gpt-4o-mini",
                                        "gpt-4o"] = "gpt-4o-mini",
        openai_embedding_model: Literal[
            "text-embedding-3-small", "text-embedding-3-large"
        ] = "text-embedding-3-small",
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
            gemini_api_key (str, optional): API key for Google Gemini. Defaults to None
            pinecone_api_key (str, optional): API key for Pinecone. Defaults to None
            pinecone_index_name (str, optional): Pinecone index name. Defaults to None
            code_interpreter (bool, optional): Enable code interpretation. Defaults to True
            openai_assistant_model (Literal["gpt-4o-mini", "gpt-4o"], optional): OpenAI model for assistant. Defaults to "gpt-4o-mini"
            openai_embedding_model (Literal["text-embedding-3-small", "text-embedding-3-large"], optional): OpenAI model for text embedding. Defaults to "text-embedding-3-small"

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
            - Optional integrations for Zep, Perplexity, Grok, Gemini, Pinecone, and Cohere
            - Supports code interpretation and custom tool functions
            - You must create the Pinecone index in the dashboard before using it
        """
        self._client = OpenAI(api_key=openai_api_key)
        self._name = name
        self._instructions = instructions
        self._openai_assistant_model = openai_assistant_model
        self._openai_embedding_model = openai_embedding_model
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
        self._pinecone_index = (
            self._pinecone.Index(
                self._pinecone_index_name) if self._pinecone else None
        )

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
            await self._database.delete_all_threads()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Perform any cleanup actions here
        pass

    async def _create_thread(self, user_id: str) -> str:
        thread_id = await self._database.get_thread_id(user_id)

        if thread_id is None:
            thread = self._client.beta.threads.create()
            thread_id = thread.id
            await self._database.save_thread_id(user_id, thread_id)
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

    # converter tool - has to be sync
    def csv_to_json(self, file_path: str) -> str:
        """Convert CSV file to JSON string format.

        Args:
            file_path (str): Path to the CSV file to convert

        Returns:
            str: JSON string containing the CSV data

        Example:
            ```python
            result = ai.csv_to_json("data.csv")
            # Returns: '[{"column1": "value1", "column2": "value2"}]'
            ```

        Note:
            This is a synchronous tool method required for OpenAI function calling.
        """
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        return json.dumps(records)

    # search kb tool - has to be sync
    def search_kb(self, query: str, limit: int = 10) -> str:
        """Search Pinecone knowledge base using OpenAI embeddings.

        Args:
            query (str): Search query to find relevant documents
            limit (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            str: JSON string of matched documents or error message

        Example:
            ```python
            results = ai.search_kb("machine learning basics", limit=5)
            # Returns: '[{"title": "ML Intro", "content": "..."}]'
            ```

        Note:
            - Requires configured Pinecone index
            - Uses OpenAI embeddings for semantic search
            - Returns JSON-serialized Pinecone match metadata results
            - Returns error message string if search fails
        """
        try:
            response = self._client.embeddings.create(
                input=query,
                model=self._openai_embedding_model,
            )
            search_results = self._pinecone_index.query(
                vector=response.data[0].embedding,
                top_k=limit,
                include_metadata=True,
                include_values=False,
            )
            matches = search_results.matches
            metadata = [match.metadata for match in matches]
            return json.dumps(metadata)
        except Exception as e:
            return f"Failed to search KB. Error: {e}"

    # add document to kb tool - has to be sync
    def add_document_to_kb(self, document: Dict[str, str]):
        """Add a document to the Pinecone knowledge base with OpenAI embeddings.

        Args:
            document (Dict[str, str]): Document to add, with string fields as values

        Example:
            ```python
            ai.add_document_to_kb({
                "title": "AI Basics",
                "content": "Introduction to artificial intelligence...",
                "author": "John Doe"
            })
            ```

        Note:
            - Requires Pinecone index to be configured
            - Uses OpenAI embeddings API
            - Document values must be strings
            - Automatically generates UUID for document
        """
        values: List[str] = []
        for _, v in document.items():
            values.append(v)
        response = self._client.embeddings.create(
            input=values,
            model=self._openai_embedding_model,
        )
        self._pinecone_index.upsert(
            vectors=[
                {
                    "id": uuid.uuid4().hex,
                    "values": response.data[0].embedding,
                    "metadata": document,
                }
            ]
        )

    # summarize tool - has to be sync
    def summarize(
        self,
        text: str,
        model: Literal["gemini-2.0-flash",
                       "gemini-1.5-pro"] = "gemini-1.5-pro",
    ) -> str:
        """Summarize text using Google's Gemini language model.

        Args:
            text (str): The text content to be summarized
            model (Literal["gemini-2.0-flash", "gemini-1.5-pro"], optional):
                Gemini model to use. Defaults to "gemini-1.5-pro"
                - gemini-2.0-flash: Faster, shorter summaries
                - gemini-1.5-pro: More detailed summaries

        Returns:
            str: Summarized text or error message if summarization fails

        Example:
            ```python
            summary = ai.summarize("Long article text here...", model="gemini-1.5-pro")
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
                        "content": "You summarize the text.",
                    },
                    {"role": "user", "content": text},
                ],
            )

            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to summarize text. Error: {e}"

    # search facts tool - has to be sync
    def search_facts(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[str] | None:
        """Search stored conversation facts using Zep memory integration.

        Args:
            user_id (str): Unique identifier for the user
            query (str): Search query to find relevant facts
            limit (int, optional): Maximum number of facts to return. Defaults to 10.

        Returns:
            List[str] | None: List of found facts or None if Zep is not configured

        Example:
            ```python
            facts = ai.search_facts(
                user_id="user123",
                query="project requirements"
            )
            # Returns: ["Fact 1", "Fact 2", ...]
            ```

        Note:
            Requires Zep integration to be configured with valid API key.
            This is a synchronous tool method required for OpenAI function calling.
        """
        if self._sync_zep:
            facts = []
            results = self._sync_zep.memory.search_sessions(
                user_id=user_id,
                text=query,
                limit=limit,
            )
            for result in results.results:
                fact = result.fact.fact
                if fact:
                    facts.append(fact)
            return facts
        return None

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
        use_perplexity: bool = True,
        use_grok: bool = True,
        use_facts: bool = True,
        use_kb=True,
        perplexity_model: Literal[
            "sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning"
        ] = "sonar",
        openai_model: Literal["o1", "o3-mini"] = "o3-mini",
        grok_model: Literal["grok-beta"] = "grok-beta",
    ) -> str:
        """Combine multiple data sources with AI reasoning to answer queries.

        Args:
            user_id (str): Unique identifier for the user
            query (str): The question or query to reason about
            use_perplexity (bool, optional): Include Perplexity search results. Defaults to True
            use_grok (bool, optional): Include X/Twitter search results. Defaults to True
            use_facts (bool, optional): Include stored conversation facts. Defaults to True
            use_kb (bool, optional): Include Pinecone knowledge base search results. Defaults to True
            perplexity_model (Literal, optional): Perplexity model to use. Defaults to "sonar"
            openai_model (Literal, optional): OpenAI model for reasoning. Defaults to "o3-mini"
            grok_model (Literal, optional): Grok model for X search. Defaults to "grok-beta"

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
            if use_facts:
                facts = self.search_facts(user_id, query)
                if not facts:
                    facts = ""
            else:
                facts = ""
            if use_perplexity:
                search_results = self.search_internet(query, perplexity_model)
            else:
                search_results = ""
            if use_grok:
                x_search_results = self.search_x(query, grok_model)
            else:
                x_search_results = ""
            if use_kb:
                kb_results = self.search_kb(query)
            else:
                kb_results = ""

            response = self._client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You combine the data with your reasoning to answer the query.",
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
    def search_x(self, query: str, model: Literal["grok-beta"] = "grok-beta") -> str:
        try:
            """Search X (formerly Twitter) using Grok API integration.

            Args:
                query (str): Search query to find relevant X posts
                model (Literal["grok-beta"], optional): Grok model to use. Defaults to "grok-beta"

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

        thread_id = await self._database.get_thread_id(user_id)

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
            "timestamp": datetime.now(),
        }

        await self._database.save_message(user_id, metadata)
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

        thread_id = await self._database.get_thread_id(user_id)

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
            "timestamp": datetime.now(),
        }

        await self._database.save_message(user_id, metadata)

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
