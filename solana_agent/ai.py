import asyncio
from datetime import datetime
import json
from typing import AsyncGenerator, List, Literal, Optional, Dict, Any, Callable
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
import aiosqlite
from openai import AssistantEventHandler
from openai.types.beta.threads import TextDelta, Text
from typing_extensions import override
import sqlite3
import inspect
import requests
from zep_python.client import AsyncZep
from zep_python.client import Zep
from zep_python.types import Message, RoleType
import pandas as pd


def adapt_datetime(ts):
    return ts.isoformat()


# Custom converter for datetime
def convert_datetime(ts):
    return datetime.fromisoformat(ts)


# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)


class EventHandler(AssistantEventHandler):
    def __init__(self, tool_handlers, ai_instance):
        super().__init__()
        self._tool_handlers = tool_handlers
        self._ai_instance = ai_instance

    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text):
        asyncio.create_task(
            self._ai_instance.accumulated_value_queue.put(delta.value))

    @override
    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self._ai_instance.handle_requires_action(event.data, run_id)


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class MongoDatabase:
    def __init__(self, db_url: str, db_name: str):
        self._client = AsyncIOMotorClient(db_url)
        self._db = self.client[db_name]
        self._threads = self.db["threads"]
        self._messages = self.db["messages"]

    async def save_thread_id(self, user_id: str, thread_id: str):
        await self._threads.insert_one({"thread_id": thread_id, "user_id": user_id})

    async def get_thread_id(self, user_id: str) -> Optional[str]:
        document = await self._threads.find_one({"user_id": user_id})
        return document["thread_id"] if document else None

    async def save_message(self, user_id: str, metadata: Dict[str, Any]):
        metadata["user_id"] = user_id
        await self._messages.insert_one(metadata)

    async def delete_all_threads(self):
        await self._threads.delete_many({})


class SQLiteDatabase:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS threads (user_id TEXT, thread_id TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (user_id TEXT, message TEXT, response TEXT, timestamp TEXT)"
        )
        self._conn.commit()
        self._conn.close()

    async def save_thread_id(self, user_id: str, thread_id: str):
        async with aiosqlite.connect(
            self._db_path, detect_types=sqlite3.PARSE_DECLTYPES
        ) as db:
            await db.execute(
                "INSERT INTO threads (user_id, thread_id) VALUES (?, ?)",
                (user_id, thread_id),
            )
            await db.commit()

    async def get_thread_id(self, user_id: str) -> Optional[str]:
        async with aiosqlite.connect(
            self._db_path, detect_types=sqlite3.PARSE_DECLTYPES
        ) as db:
            async with db.execute(
                "SELECT thread_id FROM threads WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

    async def save_message(self, user_id: str, metadata: Dict[str, Any]):
        async with aiosqlite.connect(
            self._db_path, detect_types=sqlite3.PARSE_DECLTYPES
        ) as db:
            await db.execute(
                "INSERT INTO messages (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)",
                (
                    user_id,
                    metadata["message"],
                    metadata["response"],
                    metadata["timestamp"],
                ),
            )
            await db.commit()

    async def delete_all_threads(self):
        async with aiosqlite.connect(
            self._db_path, detect_types=sqlite3.PARSE_DECLTYPES
        ) as db:
            await db.execute("DELETE FROM threads")
            await db.commit()


class AI:
    def __init__(
        self,
        openai_api_key: str,
        name: str,
        instructions: str,
        database: Any,
        zep_api_key: str = None,
        zep_base_url: str = None,
        perplexity_api_key: str = None,
        grok_api_key: str = None,
        gemini_api_key: str = None,
        code_interpreter: bool = True,
        model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini",
    ):
        self._client = OpenAI(api_key=openai_api_key)
        self._name = name
        self._instructions = instructions
        self._model = model
        self._tools = [{"type": "code_interpreter"}
                       ] if code_interpreter else []
        self._tool_handlers = {}
        self._assistant_id = None
        self._database = database
        self._accumulated_value_queue = asyncio.Queue()
        self._zep = (
            AsyncZep(api_key=zep_api_key, base_url=zep_base_url)
            if zep_api_key
            else None
        )
        self._sync_zep = (
            Zep(api_key=zep_api_key, base_url=zep_base_url) if zep_api_key else None
        )
        self._perplexity_api_key = perplexity_api_key
        self._grok_api_key = grok_api_key
        self._gemini_api_key = gemini_api_key

    async def __aenter__(self):
        assistants = self._client.beta.assistants.list()
        existing_assistant = next(
            (a for a in assistants if a.name == self._name), None)

        if existing_assistant:
            self._assistant_id = existing_assistant.id
        else:
            self._assistant_id = self._client.beta.assistants.create(
                name=self.name,
                instructions=self._instructions,
                tools=self._tools,
                model=self._model,
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
                await self._zep.user.add(user_id=user_id)
                await self._zep.memory.add_session(user_id=user_id, session_id=user_id)

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
            thread_id=thread_id, run_id=run_id)
        return run.status

    # converter tool - has to be sync
    def csv_to_json(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        return json.dumps(records)

    # summarize tool - has to be sync
    def summarize(
        self, text: str, model: Literal["gemini-2.0-flash", "gemini-1.5-pro"] = "gemini-1.5-pro"
    ) -> str:
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
        limit: int | None = None,
    ) -> List[str] | None:
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
        perplexity_model: Literal[
            "sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning"
        ] = "sonar",
        openai_model: Literal["o1", "o3-mini"] = "o3-mini",
        grok_model: Literal["grok-beta"] = "grok-beta",
    ) -> str:
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

            response = self._client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You combine the data with your reasoning to answer the query.",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}, Facts: {facts}, Internet Search Results: {search_results}, X Search Results: {x_search_results}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to reason. Error: {e}"

    # x search tool - has to be sync
    def search_x(self, query: str, model: Literal["grok-beta"] = "grok-beta") -> str:
        try:
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
        if self._zep:
            await self._zep.memory.delete(session_id=user_id)

    async def listen(self, audio_content: bytes, input_format: str) -> str:
        transcription = self._client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"file.{input_format}", audio_content),
        )
        return transcription.text

    async def text(self, user_id: str, user_text: str) -> AsyncGenerator[str, None]:
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
                    role_type=RoleType["user"],
                    content=user_text,
                ),
                Message(
                    role="assistant",
                    role_type=RoleType["assistant"],
                    content=full_response,
                ),
            ]
            await self._zep.memory.add(
                user_id=user_id, session_id=user_id, messages=messages
            )

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
        # Reset the queue for each new conversation
        self._accumulated_value_queue = asyncio.Queue()

        thread_id = await self._database.get_thread_id(user_id)

        if thread_id is None:
            thread_id = await self._create_thread(user_id)

        self._current_thread_id = thread_id
        transcript = await self.listen(audio_bytes, input_format)
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
                    role_type=RoleType["user"],
                    content=transcript,
                ),
                Message(
                    role="assistant",
                    role_type=RoleType["assistant"],
                    content=full_response,
                ),
            ]
            await self._zep.memory.add(
                user_id=user_id, session_id=user_id, messages=messages
            )

        # Generate and stream the audio response
        with self._client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=full_response,
            response_format=response_format,
        ) as response:
            for chunk in response.iter_bytes(1024):
                yield chunk

    def handle_requires_action(self, data, run_id):
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
