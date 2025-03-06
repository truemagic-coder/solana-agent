"""
Solana Agent System: AI-powered agent coordination system with human agent integration.

This module implements a clean architecture approach with:
- Domain models for core data structures
- Interfaces for dependency inversion
- Services for business logic
- Repositories for data access
- Adapters for external integrations
- Use cases for orchestrating application flows
"""
import asyncio
import datetime
import json
import re
import traceback
import uuid
from enum import Enum
from typing import (AsyncGenerator, Callable, Dict, List, Literal, Optional,
                    Protocol, Tuple, Any)
from pydantic import BaseModel, Field
from pymongo import MongoClient
from openai import OpenAI
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_python.client import AsyncZep
from zep_cloud.types import Message
from pinecone import Pinecone


#############################################
# DOMAIN MODELS
#############################################

class TicketStatus(str, Enum):
    """Represents possible states of a support ticket."""
    NEW = "new"
    ACTIVE = "active"
    PENDING = "pending"
    TRANSFERRED = "transferred"
    RESOLVED = "resolved"
    PLANNING = "planning"


class AgentType(str, Enum):
    """Type of agent (AI or Human)."""
    AI = "ai"
    HUMAN = "human"


class Ticket(BaseModel):
    """Represents a user support ticket."""
    id: str
    user_id: str
    query: str
    status: TicketStatus
    assigned_to: str
    created_at: datetime.datetime
    complexity: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    is_parent: bool = False
    is_subtask: bool = False
    parent_id: Optional[str] = None
    updated_at: Optional[datetime.datetime] = None
    resolved_at: Optional[datetime.datetime] = None
    handoff_reason: Optional[str] = None


class Handoff(BaseModel):
    """Represents a ticket handoff between agents."""
    ticket_id: str
    user_id: str
    from_agent: str
    to_agent: str
    reason: str
    query: str
    timestamp: datetime.datetime
    automatic: bool = False


class NPSSurvey(BaseModel):
    """Represents an NPS survey for a resolved ticket."""
    survey_id: str
    user_id: str
    ticket_id: str
    agent_name: str
    status: str
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    score: Optional[int] = None
    feedback: Optional[str] = None


class MemoryInsight(BaseModel):
    """Factual insight extracted from user conversations."""
    fact: str = Field(...,
                      description="The factual information worth remembering")
    relevance: str = Field(...,
                           description="Short explanation of why this fact is generally useful")


class TicketResolution(BaseModel):
    """Information about ticket resolution status."""
    status: Literal["resolved", "needs_followup", "cannot_determine"] = Field(
        ..., description="Resolution status of the ticket"
    )
    confidence: float = Field(
        ..., description="Confidence score for the resolution decision (0.0-1.0)")
    reasoning: str = Field(...,
                           description="Brief explanation for the resolution decision")
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested follow-up actions if needed")


class EscalationRequirements(BaseModel):
    """Information about requirements for escalation to human agents."""
    has_sufficient_info: bool = Field(
        ..., description="Whether enough information has been collected for escalation")
    missing_fields: List[str] = Field(
        default_factory=list, description="Required fields that are missing")
    recommendation: str = Field(...,
                                description="Recommendation for next steps")


class ImprovementArea(BaseModel):
    """Area for improvement identified by the critic."""
    area: str = Field(...,
                      description="Area name (e.g., 'Accuracy', 'Completeness')")
    issue: str = Field(..., description="Specific issue identified")
    recommendation: str = Field(...,
                                description="Specific actionable improvement")


class CritiqueFeedback(BaseModel):
    """Comprehensive feedback from critic review."""
    strengths: List[str] = Field(
        default_factory=list, description="List of strengths in the response")
    improvement_areas: List[ImprovementArea] = Field(
        default_factory=list, description="Areas needing improvement")
    overall_score: float = Field(..., description="Score between 0.0 and 1.0")
    priority: Literal["low", "medium",
                      "high"] = Field(..., description="Priority level for improvements")


class NPSResponse(BaseModel):
    """User response to an NPS survey."""
    score: int = Field(..., ge=0, le=10, description="NPS score (0-10)")
    feedback: str = Field("", description="Optional feedback comment")
    improvement_suggestions: str = Field(
        "", description="Suggestions for improvement")


class CollectiveMemoryResponse(BaseModel):
    """Response format for collective memory extraction."""
    insights: List[MemoryInsight] = Field(
        default_factory=list, description="List of factual insights extracted")


class DocumentModel(BaseModel):
    """Document for knowledge base storage."""
    id: str
    text: str


class AgentScore(BaseModel):
    """Comprehensive performance score for an agent."""
    agent_name: str
    overall_score: float
    rating: str
    components: Dict[str, float]
    metrics: Dict[str, Any]
    period: Dict[str, str]


class PlanStatus(BaseModel):
    """Status information for a complex task plan."""
    visualization: str
    progress: int
    status: str
    estimated_completion: str
    subtask_count: int


#############################################
# INTERFACES
#############################################

class LLMProvider(Protocol):
    """Interface for language model providers."""

    async def generate_text(
        self,
        user_id: str,
        prompt: str,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]: ...

    def generate_embedding(self, text: str) -> List[float]: ...


class MemoryProvider(Protocol):
    """Interface for conversation memory providers."""

    async def store(self, user_id: str,
                    messages: List[Dict[str, Any]]) -> None: ...

    async def retrieve(self, user_id: str) -> str: ...

    async def delete(self, user_id: str) -> None: ...


class VectorStoreProvider(Protocol):
    """Interface for vector storage providers."""

    def store_vectors(self, vectors: List[Dict], namespace: str) -> None: ...

    def search_vectors(
        self, query_vector: List[float], namespace: str, limit: int = 5) -> List[Dict]: ...

    def delete_vector(self, id: str, namespace: str) -> None: ...


class DataStorageProvider(Protocol):
    """Interface for data storage providers."""

    def create_collection(self, name: str) -> None: ...

    def collection_exists(self, name: str) -> bool: ...

    def insert_one(self, collection: str, document: Dict) -> str: ...

    def find_one(self, collection: str, query: Dict) -> Optional[Dict]: ...

    def find(self, collection: str, query: Dict,
             sort: Optional[List[Tuple]] = None, limit: int = 0) -> List[Dict]: ...

    def update_one(self, collection: str, query: Dict,
                   update: Dict) -> bool: ...

    def delete_one(self, collection: str, query: Dict) -> bool: ...

    def count_documents(self, collection: str, query: Dict) -> int: ...

    def aggregate(self, collection: str,
                  pipeline: List[Dict]) -> List[Dict]: ...

    def create_index(self, collection: str,
                     keys: List[Tuple], **kwargs) -> None: ...


class TicketRepository(Protocol):
    """Interface for ticket data access."""

    def create(self, ticket: Ticket) -> str: ...

    def get_by_id(self, ticket_id: str) -> Optional[Ticket]: ...

    def get_active_for_user(self, user_id: str) -> Optional[Ticket]: ...

    def find(self, query: Dict,
             sort_by: Optional[str] = None, limit: int = 0) -> List[Ticket]: ...

    def update(self, ticket_id: str, updates: Dict[str, Any]) -> bool: ...

    def count(self, query: Dict) -> int: ...


class HandoffRepository(Protocol):
    """Interface for handoff data access."""

    def record(self, handoff: Handoff) -> str: ...

    def find_for_agent(self, agent_name: str, start_date: Optional[datetime.datetime] = None,
                       end_date: Optional[datetime.datetime] = None) -> List[Handoff]: ...

    def count_for_agent(self, agent_name: str, start_date: Optional[datetime.datetime] = None,
                        end_date: Optional[datetime.datetime] = None) -> int: ...


class NPSSurveyRepository(Protocol):
    """Interface for NPS survey data access."""

    def create(self, survey: NPSSurvey) -> str: ...

    def get_by_id(self, survey_id: str) -> Optional[NPSSurvey]: ...

    def update_response(self, survey_id: str, score: int,
                        feedback: Optional[str] = None) -> bool: ...

    def get_metrics(self, agent_name: Optional[str] = None, start_date: Optional[datetime.datetime] = None,
                    end_date: Optional[datetime.datetime] = None) -> Dict[str, Any]: ...


class MemoryRepository(Protocol):
    """Interface for collective memory data access."""

    def store_insight(self, user_id: str, insight: MemoryInsight) -> str: ...

    def search(self, query: str, limit: int = 5) -> List[Dict]: ...


class AgentRegistry(Protocol):
    """Interface for agent management."""

    def register_ai_agent(self, name: str, agent: Any,
                          specialization: str) -> None: ...

    def register_human_agent(self, agent_id: str, name: str, specialization: str,
                             notification_handler: Optional[Callable] = None) -> Any: ...

    def get_ai_agent(self, name: str) -> Optional[Any]: ...

    def get_human_agent(self, agent_id: str) -> Optional[Any]: ...

    def get_all_ai_agents(self) -> Dict[str, Any]: ...

    def get_all_human_agents(self) -> Dict[str, Any]: ...

    def get_specializations(self) -> Dict[str, str]: ...


#############################################
# IMPLEMENTATIONS - ADAPTERS
#############################################

class MongoDBAdapter:
    """MongoDB implementation of DataStorageProvider."""

    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]

    def create_collection(self, name: str) -> None:
        if name not in self.db.list_collection_names():
            self.db.create_collection(name)

    def collection_exists(self, name: str) -> bool:
        return name in self.db.list_collection_names()

    def insert_one(self, collection: str, document: Dict) -> str:
        if "_id" not in document:
            document["_id"] = str(uuid.uuid4())
        self.db[collection].insert_one(document)
        return document["_id"]

    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        return self.db[collection].find_one(query)

    def find(self, collection: str, query: Dict, sort: Optional[List[Tuple]] = None, limit: int = 0) -> List[Dict]:
        cursor = self.db[collection].find(query)
        if sort:
            cursor = cursor.sort(sort)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)

    def update_one(self, collection: str, query: Dict, update: Dict) -> bool:
        result = self.db[collection].update_one(query, update)
        return result.modified_count > 0

    def delete_one(self, collection: str, query: Dict) -> bool:
        result = self.db[collection].delete_one(query)
        return result.deleted_count > 0

    def count_documents(self, collection: str, query: Dict) -> int:
        return self.db[collection].count_documents(query)

    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        return list(self.db[collection].aggregate(pipeline))

    def create_index(self, collection: str, keys: List[Tuple], **kwargs) -> None:
        self.db[collection].create_index(keys, **kwargs)


class OpenAIAdapter:
    """OpenAI implementation of LLMProvider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def generate_text(
        self,
        user_id: str,
        prompt: str,
        system_prompt: str = "",
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from OpenAI models with streaming."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            stream=stream,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", None),
        )

        if stream:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield response.choices[0].message.content

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


class ZepMemoryAdapter:
    """Zep implementation of MemoryProvider."""

    def __init__(self, api_key: str = None, base_url: str = None):
        if api_key and not base_url:
            # Cloud version
            self.client = AsyncZepCloud(api_key=api_key)
        elif api_key and base_url:
            # Self-hosted version with authentication
            self.client = AsyncZep(api_key=api_key, base_url=base_url)
        else:
            # Self-hosted version without authentication
            self.client = AsyncZep(base_url="http://localhost:8000")

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in Zep memory."""
        zep_messages = [
            Message(
                role=msg["role"],
                role_type=msg["role"],
                content=self._truncate(msg["content"], 2500),
            )
            for msg in messages
        ]
        await self.client.memory.add(session_id=user_id, messages=zep_messages)

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context for a user."""
        try:
            memory = await self.client.memory.get_session(user_id)
            summary = await self.client.memory.summarize(user_id)

            # Format the memory context
            context = f"Summary: {summary.summary}\n\n"

            # Add most relevant facts if available
            if hasattr(memory, "metadata") and memory.metadata and "facts" in memory.metadata:
                facts = memory.metadata["facts"]
                if facts:
                    context += "Key facts:\n"
                    for fact in facts[:5]:  # Limit to top 5 facts
                        context += f"- {fact['fact']}\n"

            return context
        except Exception as e:
            return f"Error retrieving memory: {e}"

    async def delete(self, user_id: str) -> None:
        """Delete memory for a user."""
        try:
            await self.client.memory.delete(session_id=user_id)
            await self.client.user.delete(user_id=user_id)
        except Exception as e:
            print(f"Error deleting memory: {e}")

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if last_period > limit * 0.8:  # Only use period if it's reasonably close to the end
            return truncated[:last_period + 1]

        return truncated + "..."


class PineconeAdapter:
    """Pinecone implementation of VectorStoreProvider."""

    def __init__(self, api_key: str, index_name: str, embedding_model: str = "llama-text-embed-v2"):
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.embedding_model = embedding_model

    def store_vectors(self, vectors: List[Dict], namespace: str) -> None:
        """Store vectors in Pinecone."""
        self.index.upsert(vectors=vectors, namespace=namespace)

    def search_vectors(self, query_vector: List[float], namespace: str, limit: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        results = self.index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            namespace=namespace
        )

        # Format results
        output = []
        if hasattr(results, "matches"):
            for match in results.matches:
                if hasattr(match, "metadata") and match.metadata:
                    output.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    })

        return output

    def delete_vector(self, id: str, namespace: str) -> None:
        """Delete a vector by ID."""
        self.index.delete(ids=[id], namespace=namespace)


#############################################
# IMPLEMENTATIONS - REPOSITORIES
#############################################

class MongoTicketRepository:
    """MongoDB implementation of TicketRepository."""

    def __init__(self, db_provider: DataStorageProvider):
        self.db = db_provider
        self.collection = "tickets"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("user_id", 1)])
        self.db.create_index(self.collection, [("status", 1)])
        self.db.create_index(self.collection, [("assigned_to", 1)])

    def create(self, ticket: Ticket) -> str:
        """Create a new ticket."""
        return self.db.insert_one(self.collection, ticket.model_dump(mode="json"))

    def get_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID."""
        data = self.db.find_one(self.collection, {"_id": ticket_id})
        return Ticket(**data) if data else None

    def get_active_for_user(self, user_id: str) -> Optional[Ticket]:
        """Get active ticket for a user."""
        data = self.db.find_one(
            self.collection,
            {
                "user_id": user_id,
                "status": {"$in": ["new", "active", "pending", "transferred"]}
            }
        )
        return Ticket(**data) if data else None

    def find(self, query: Dict, sort_by: Optional[str] = None, limit: int = 0) -> List[Ticket]:
        """Find tickets matching query."""
        sort_params = [(sort_by, 1)] if sort_by else [("created_at", -1)]
        data = self.db.find(self.collection, query, sort_params, limit)
        return [Ticket(**item) for item in data]

    def update(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update a ticket."""
        return self.db.update_one(
            self.collection,
            {"_id": ticket_id},
            {"$set": updates}
        )

    def count(self, query: Dict) -> int:
        """Count tickets matching query."""
        return self.db.count_documents(self.collection, query)


class MongoHandoffRepository:
    """MongoDB implementation of HandoffRepository."""

    def __init__(self, db_provider: DataStorageProvider):
        self.db = db_provider
        self.collection = "handoffs"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("from_agent", 1)])
        self.db.create_index(self.collection, [("to_agent", 1)])
        self.db.create_index(self.collection, [("timestamp", 1)])

    def record(self, handoff: Handoff) -> str:
        """Record a new handoff."""
        return self.db.insert_one(self.collection, handoff.model_dump(mode="json"))

    def find_for_agent(
        self,
        agent_name: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> List[Handoff]:
        """Find handoffs for an agent."""
        query = {"from_agent": agent_name}

        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        data = self.db.find(self.collection, query)
        return [Handoff(**item) for item in data]

    def count_for_agent(
        self,
        agent_name: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> int:
        """Count handoffs for an agent."""
        query = {"from_agent": agent_name}

        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        return self.db.count_documents(self.collection, query)


class MongoNPSSurveyRepository:
    """MongoDB implementation of NPSSurveyRepository."""

    def __init__(self, db_provider: DataStorageProvider):
        self.db = db_provider
        self.collection = "nps_surveys"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("survey_id", 1)])
        self.db.create_index(self.collection, [("agent_name", 1)])
        self.db.create_index(self.collection, [("status", 1)])

    def create(self, survey: NPSSurvey) -> str:
        """Create a new NPS survey."""
        return self.db.insert_one(self.collection, survey.model_dump(mode="json"))

    def get_by_id(self, survey_id: str) -> Optional[NPSSurvey]:
        """Get a survey by ID."""
        data = self.db.find_one(self.collection, {"survey_id": survey_id})
        return NPSSurvey(**data) if data else None

    def update_response(self, survey_id: str, score: int, feedback: Optional[str] = None) -> bool:
        """Update a survey with user response."""
        updates = {
            "score": score,
            "status": "completed",
            "completed_at": datetime.datetime.now(datetime.timezone.utc)
        }

        if feedback:
            updates["feedback"] = feedback

        return self.db.update_one(
            self.collection,
            {"survey_id": survey_id},
            {"$set": updates}
        )

    def get_metrics(
        self,
        agent_name: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """Get NPS metrics."""
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

        # Get responses
        responses = self.db.find(self.collection, query)

        if not responses:
            return {
                "nps_score": 0,
                "promoters": 0,
                "passives": 0,
                "detractors": 0,
                "total_responses": 0,
                "avg_score": 0,
            }

        # Count categories
        promoters = sum(1 for r in responses if r.get("score", 0) >= 9)
        passives = sum(1 for r in responses if 7 <= r.get("score", 0) <= 8)
        detractors = sum(1 for r in responses if r.get("score", 0) <= 6)

        total = len(responses)

        # Calculate NPS
        nps_score = int(((promoters - detractors) / total) * 100)

        # Calculate average score
        avg_score = sum(r.get("score", 0) for r in responses) / total

        # Get agent breakdown if no specific agent was requested
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
            for result in self.db.aggregate(self.collection, pipeline):
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


class MongoMemoryRepository:
    """MongoDB implementation of MemoryRepository."""

    def __init__(self, db_provider: DataStorageProvider, vector_provider: Optional[VectorStoreProvider] = None):
        self.db = db_provider
        self.vector_db = vector_provider
        self.collection = "collective_memory"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes for text search
        try:
            self.db.create_index(
                self.collection,
                [("fact", "text"), ("relevance", "text")]
            )
        except Exception as e:
            print(f"Warning: Text index creation might have failed: {e}")

    def store_insight(self, user_id: str, insight: MemoryInsight) -> str:
        """Store a new insight in memory."""
        record_id = str(uuid.uuid4())
        record = {
            "_id": record_id,
            "fact": insight.fact,
            "relevance": insight.relevance,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "source_user_id": user_id,
        }

        # Store in MongoDB
        self.db.insert_one(self.collection, record)

        # Store in vector DB if available
        if self.vector_db:
            try:
                text = f"{insight.fact}: {insight.relevance}"
                embedding = self._get_embedding_for_text(text)

                if embedding:
                    vector = {
                        "id": record_id,
                        "values": embedding,
                        "metadata": {
                            "fact": insight.fact,
                            "relevance": insight.relevance,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "source_user_id": user_id,
                        },
                    }
                    self.vector_db.store_vectors([vector], namespace="memory")
            except Exception as e:
                print(f"Error storing vector: {e}")

            return record_id

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights."""
        results = []

        # Try vector search first if available
        if self.vector_db:
            try:
                embedding = self._get_embedding_for_text(query)
                if embedding:
                    vector_results = self.vector_db.search_vectors(
                        embedding, namespace="memory", limit=limit
                    )
                    for result in vector_results:
                        results.append({
                            "id": result["id"],
                            "fact": result["metadata"]["fact"],
                            "relevance": result["metadata"]["relevance"],
                            "similarity": result["score"]
                        })
                    return results
            except Exception as e:
                print(f"Error in vector search: {e}")

        # Fall back to text search
        try:
            query_dict = {"$text": {"$search": query}}
            mongo_results = self.db.find(
                self.collection,
                query_dict,
                [("score", {"$meta": "textScore"})],
                limit
            )

            for doc in mongo_results:
                results.append({
                    "id": doc["_id"],
                    "fact": doc["fact"],
                    "relevance": doc["relevance"],
                    "timestamp": doc["timestamp"].isoformat() if isinstance(
                        doc["timestamp"], datetime.datetime) else doc["timestamp"]
                })
            return results
        except Exception as e:
            print(f"Error in text search: {e}")
            return []

    def _get_embedding_for_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI."""
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None


#############################################
# SERVICES
#############################################

class RoutingService:
    """Service for routing queries to appropriate agents."""

    def __init__(self, llm_provider: LLMProvider, agent_registry: AgentRegistry, router_model: str = "gpt-4o-mini"):
        self.llm_provider = llm_provider
        self.agent_registry = agent_registry
        self.router_model = router_model

    async def route_query(self, query: str) -> str:
        """Route query to the most appropriate AI agent."""
        specializations = self.agent_registry.get_specializations()
        # Get AI-only specializations
        ai_specialists = {k: v for k, v in specializations.items()
                          if k in self.agent_registry.get_all_ai_agents()}

        # Create routing prompt
        prompt = f"""
        Analyze this user query and return the MOST APPROPRIATE AI specialist.
        
        User query: "{query}"
        
        Available AI specialists:
        {json.dumps(ai_specialists, indent=2)}
        
        CRITICAL INSTRUCTIONS:
        1. Choose specialists based on domain expertise match.
        2. Return EXACTLY ONE specialist name.
        """

        # Generate routing decision
        response = ""
        async for chunk in self.llm_provider.generate_text(
            "router",
            prompt,
            system_prompt="You are a routing system that matches queries to the best specialist.",
            stream=False,
            model=self.router_model,
            temperature=0.2
        ):
            response += chunk

        # Match to an actual agent name
        agent_name = self._match_agent_name(
            response.strip(), list(ai_specialists.keys()))

        return agent_name

    def _match_agent_name(self, response: str, agent_names: List[str]) -> str:
        """Match router response to an actual AI agent name."""
        # Exact match (priority)
        if response in agent_names:
            return response

        # Case-insensitive match
        for name in agent_names:
            if name.lower() == response.lower():
                return name

        # Partial match
        for name in agent_names:
            if name.lower() in response.lower():
                return name

        # Fallback to first AI agent
        return agent_names[0] if agent_names else "default"


class TicketService:
    """Service for managing tickets and their lifecycle."""

    def __init__(self, ticket_repository: TicketRepository):
        self.ticket_repository = ticket_repository

    async def get_or_create_ticket(self, user_id: str, query: str, complexity: Optional[Dict[str, Any]] = None) -> Ticket:
        """Get active ticket for user or create a new one."""
        # Check for active ticket
        ticket = self.ticket_repository.get_active_for_user(user_id)
        if ticket:
            return ticket

        # Create new ticket
        new_ticket = Ticket(
            id=str(uuid.uuid4()),
            user_id=user_id,
            query=query,
            status=TicketStatus.NEW,
            assigned_to="",  # Will be assigned later
            created_at=datetime.datetime.now(datetime.timezone.utc),
            complexity=complexity
        )

        ticket_id = self.ticket_repository.create(new_ticket)
        new_ticket.id = ticket_id
        return new_ticket

    def update_ticket_status(self, ticket_id: str, status: TicketStatus, **additional_updates) -> bool:
        """Update ticket status and additional fields."""
        updates = {"status": status,
                   "updated_at": datetime.datetime.now(datetime.timezone.utc)}
        updates.update(additional_updates)

        return self.ticket_repository.update(ticket_id, updates)

    def mark_ticket_resolved(self, ticket_id: str, resolution_data: Dict[str, Any]) -> bool:
        """Mark a ticket as resolved with resolution information."""
        updates = {
            "status": TicketStatus.RESOLVED,
            "resolved_at": datetime.datetime.now(datetime.timezone.utc),
            "resolution_confidence": resolution_data.get("confidence", 0.0),
            "resolution_reasoning": resolution_data.get("reasoning", ""),
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        }

        return self.ticket_repository.update(ticket_id, updates)


class HandoffService:
    """Service for managing handoffs between agents."""

    def __init__(self,
                 handoff_repository: HandoffRepository,
                 ticket_repository: TicketRepository,
                 agent_registry: AgentRegistry):
        self.handoff_repository = handoff_repository
        self.ticket_repository = ticket_repository
        self.agent_registry = agent_registry

    async def process_handoff(self,
                              ticket_id: str,
                              from_agent: str,
                              to_agent: str,
                              reason: str) -> str:
        """Process a handoff between agents."""
        # Get ticket information
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        # Check if target agent exists
        if to_agent not in self.agent_registry.get_all_ai_agents() and \
            (not hasattr(self.agent_registry, "get_all_human_agents") or
             to_agent not in self.agent_registry.get_all_human_agents()):
            raise ValueError(f"Target agent {to_agent} not found")

        # Record the handoff
        handoff = Handoff(
            ticket_id=ticket_id,
            user_id=ticket.user_id,
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            query=ticket.query,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

        self.handoff_repository.record(handoff)

        # Update the ticket
        self.ticket_repository.update(
            ticket_id,
            {
                "assigned_to": to_agent,
                "status": TicketStatus.TRANSFERRED,
                "handoff_reason": reason,
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
            }
        )

        return to_agent


class NPSService:
    """Service for managing NPS surveys and ratings."""

    def __init__(self, nps_repository: NPSSurveyRepository, ticket_repository: TicketRepository):
        self.nps_repository = nps_repository
        self.ticket_repository = ticket_repository

    def create_survey(self, user_id: str, ticket_id: str, agent_name: str) -> str:
        """Create an NPS survey for a completed ticket."""
        survey = NPSSurvey(
            survey_id=str(uuid.uuid4()),
            user_id=user_id,
            ticket_id=ticket_id,
            agent_name=agent_name,
            status="pending",
            created_at=datetime.datetime.now(datetime.timezone.utc)
        )

        return self.nps_repository.create(survey)

    def process_response(self, survey_id: str, score: int, feedback: Optional[str] = None) -> bool:
        """Process user response to NPS survey."""
        return self.nps_repository.update_response(survey_id, score, feedback)

    def get_agent_score(self, agent_name: str, start_date=None, end_date=None) -> Dict[str, Any]:
        """Calculate a comprehensive agent score."""
        # Get NPS metrics
        nps_metrics = self.nps_repository.get_metrics(
            agent_name, start_date, end_date)

        # Get ticket metrics (assuming calculated elsewhere)
        # This is a simplified implementation - in practice we'd get more metrics
        nps_score = nps_metrics.get(
            "avg_score", 0) * 10  # Convert 0-10 to 0-100

        # Calculate overall score - simplified version
        overall_score = nps_score

        return {
            "agent_name": agent_name,
            "overall_score": round(overall_score, 1),
            "rating": self._get_score_rating(overall_score),
            "components": {
                "nps": round(nps_score, 1),
            },
            "metrics": {
                "nps_responses": nps_metrics.get("total_responses", 0),
            },
            "period": {
                "start": start_date.isoformat() if start_date else "All time",
                "end": end_date.isoformat() if end_date else "Present"
            }
        }

    def _get_score_rating(self, score: float) -> str:
        """Convert numerical score to descriptive rating."""
        if score >= 90:
            return "Outstanding"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 50:
            return "Average"
        elif score >= 40:
            return "Below Average"
        elif score >= 30:
            return "Poor"
        else:
            return "Needs Improvement"


class MemoryService:
    """Service for managing collective memory and insights."""

    def __init__(self, memory_repository: MemoryRepository, llm_provider: LLMProvider):
        self.memory_repository = memory_repository
        self.llm_provider = llm_provider

    async def extract_insights(self, user_id: str, conversation: Dict[str, str]) -> List[MemoryInsight]:
        """Extract insights from a conversation."""
        prompt = f"""
        Extract factual, generalizable insights from this conversation that would be valuable to remember.
        
        User: {conversation.get('message', '')}
        Assistant: {conversation.get('response', '')}
        
        Extract only factual information that would be useful for future similar conversations.
        Ignore subjective opinions, preferences, or greeting messages.
        Return insights as a JSON array of objects with "fact" and "relevance" fields.
        Only extract high-quality insights worth remembering.
        If no valuable insights exist, return an empty array.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            user_id,
            prompt,
            system_prompt="Extract factual insights from conversations.",
            stream=False,
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"}
        ):
            response += chunk

        # Parse response
        try:
            data = json.loads(response)
            insights = data.get("insights", [])
            return [MemoryInsight(**insight) for insight in insights]
        except Exception as e:
            print(f"Error parsing insights: {e}")
            return []

    async def store_insights(self, user_id: str, insights: List[MemoryInsight]) -> None:
        """Store multiple insights in memory."""
        for insight in insights:
            self.memory_repository.store_insight(user_id, insight)

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights."""
        return self.memory_repository.search(query, limit)


class CriticService:
    """Service for providing critique and feedback on agent responses."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.feedback_collection = []

    async def critique_response(self, user_query: str, agent_response: str, agent_name: str) -> CritiqueFeedback:
        """Analyze and critique an agent's response."""
        prompt = f"""
        Analyze this agent's response to the user query and provide detailed feedback:
        
        USER QUERY: {user_query}
        
        AGENT RESPONSE: {agent_response}
        
        Provide a structured critique with:
        1. Strengths of the response
        2. Areas for improvement with specific issues and recommendations
        3. Overall quality score (0.0-1.0)
        4. Priority level for improvements (low/medium/high)
        
        Format as JSON matching the CritiqueFeedback schema.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            "critic",
            prompt,
            system_prompt="You are an expert evaluator of AI responses. Provide objective, specific feedback.",
            stream=False,
            model="gpt-4o",
            temperature=0.2,
            response_format={"type": "json_object"}
        ):
            response += chunk

        try:
            data = json.loads(response)
            feedback = CritiqueFeedback(**data)

            # Store feedback for analytics
            self.feedback_collection.append({
                "agent_name": agent_name,
                "strengths_count": len(feedback.strengths),
                "issues_count": len(feedback.improvement_areas),
                "overall_score": feedback.overall_score,
                "priority": feedback.priority,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

            return feedback
        except Exception as e:
            print(f"Error parsing critique feedback: {e}")
            return CritiqueFeedback(
                strengths=["Unable to analyze response"],
                improvement_areas=[],
                overall_score=0.5,
                priority="medium"
            )

    def get_agent_feedback(self, agent_name: str, limit: int = 50) -> List[Dict]:
        """Get historical feedback for a specific agent."""
        return [fb for fb in self.feedback_collection if fb["agent_name"] == agent_name][-limit:]


class AgentService:
    """Service for managing AI and human agents."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.ai_agents = {}
        self.human_agents = {}
        self.specializations = {}

    def register_ai_agent(self, name: str, instructions: str, specialization: str, model: str = "gpt-4o") -> None:
        """Register an AI agent with its specialization."""
        self.ai_agents[name] = {
            "instructions": instructions,
            "model": model
        }
        self.specializations[name] = specialization

    def register_human_agent(self, agent_id: str, name: str, specialization: str, notification_handler: Optional[Callable] = None) -> None:
        """Register a human agent."""
        self.human_agents[agent_id] = {
            "name": name,
            "specialization": specialization,
            "notification_handler": notification_handler,
            "availability_status": "available"
        }
        self.specializations[agent_id] = specialization

    async def generate_response(self, agent_name: str, user_id: str, query: str, memory_context: str = "", **kwargs) -> AsyncGenerator[str, None]:
        """Generate response from an AI agent."""
        if agent_name not in self.ai_agents:
            yield "Error: Agent not found"
            return

        agent_config = self.ai_agents[agent_name]

        # Get instructions and add memory context
        instructions = agent_config["instructions"]
        if memory_context:
            instructions += f"\n\nUser context and history:\n{memory_context}"

        # Generate response
        async for chunk in self.llm_provider.generate_text(
            user_id=user_id,
            prompt=query,
            system_prompt=instructions,
            model=agent_config["model"],
            **kwargs
        ):
            yield chunk

    def get_all_ai_agents(self) -> Dict[str, Any]:
        """Get all registered AI agents."""
        return self.ai_agents

    def get_all_human_agents(self) -> Dict[str, Any]:
        """Get all registered human agents."""
        return self.human_agents

    def get_specializations(self) -> Dict[str, str]:
        """Get specializations of all agents."""
        return self.specializations

    def update_human_agent_status(self, agent_id: str, status: str) -> bool:
        """Update a human agent's availability status."""
        if agent_id in self.human_agents:
            self.human_agents[agent_id]["availability_status"] = status
            return True
        return False

#############################################
# MAIN AGENT PROCESSOR
#############################################


class QueryProcessor:
    """Main service to process user queries using agents and services."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        ticket_service: TicketService,
        handoff_service: HandoffService,
        memory_service: MemoryService,
        nps_service: NPSService,
        critic_service: Optional[CriticService] = None,
        memory_provider: Optional[MemoryProvider] = None,
        enable_critic: bool = True,
        router_model: str = "gpt-4o-mini"
    ):
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.ticket_service = ticket_service
        self.handoff_service = handoff_service
        self.memory_service = memory_service
        self.nps_service = nps_service
        self.critic_service = critic_service
        self.memory_provider = memory_provider
        self.enable_critic = enable_critic
        self.router_model = router_model
        self._shutdown_event = asyncio.Event()

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle ticket management."""
        try:
            # Handle human agent messages differently
            if await self._is_human_agent(user_id):
                async for chunk in self._process_human_agent_message(user_id, user_text):
                    yield chunk
                return

            # Handle simple greetings without full agent routing
            if await self._is_simple_greeting(user_text):
                greeting_response = await self._generate_greeting_response(user_id, user_text)
                yield greeting_response
                return

            # Handle system commands
            command_response = await self._process_system_commands(user_id, user_text)
            if command_response is not None:
                yield command_response
                return

            # Check for active ticket
            active_ticket = self.ticket_service.get_active_for_user(user_id)

            if active_ticket:
                # Process existing ticket
                async for chunk in self._process_existing_ticket(user_id, user_text, active_ticket, timezone):
                    yield chunk
            else:
                # Create new ticket
                complexity = await self._assess_task_complexity(user_text)

                # Process as new ticket
                async for chunk in self._process_new_ticket(user_id, user_text, complexity, timezone):
                    yield chunk

        except Exception as e:
            print(f"Error in request processing: {str(e)}")
            print(traceback.format_exc())
            yield "\n\nI apologize for the technical difficulty.\n\n"

    async def _is_human_agent(self, user_id: str) -> bool:
        """Check if the user is a registered human agent."""
        return user_id in self.agent_service.get_all_human_agents()

    async def _is_simple_greeting(self, text: str) -> bool:
        """Determine if the user message is a simple greeting."""
        text_lower = text.lower().strip()

        # Common greetings list
        simple_greetings = [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "what's up", "how are you", "how's it going"
        ]

        # Check if text starts with a greeting and is relatively short
        is_greeting = any(text_lower.startswith(greeting)
                          for greeting in simple_greetings)
        # Arbitrary threshold for "simple" messages
        is_short = len(text.split()) < 7

        return is_greeting and is_short

    async def _generate_greeting_response(self, user_id: str, text: str) -> str:
        """Generate a friendly response to a simple greeting."""
        # Get user context if available
        context = ""
        if self.memory_provider:
            context = await self.memory_provider.retrieve(user_id)

        # Get first available AI agent for the greeting
        first_agent_name = next(
            iter(self.agent_service.get_all_ai_agents().keys()))

        response = ""
        async for chunk in self.agent_service.generate_response(
            first_agent_name,
            user_id,
            text,
            context,
            temperature=0.7,
            max_tokens=100  # Keep it brief
        ):
            response += chunk

        # Store in memory if available
        if self.memory_provider:
            await self.memory_provider.store(
                user_id,
                [
                    {"role": "user", "content": text},
                    {"role": "assistant",
                        "content": self._truncate(response, 2500)}
                ]
            )

        return response

    async def _process_system_commands(self, user_id: str, user_text: str) -> Optional[str]:
        """Process system commands and return response if command was handled."""
        # Simple command system
        if user_text.startswith("!"):
            command_parts = user_text.split(" ", 1)
            command = command_parts[0].lower()
            args = command_parts[1] if len(command_parts) > 1 else ""

            if command == "!memory" and args:
                # Search collective memory
                results = self.memory_service.search_memory(args)

                if not results:
                    return "No relevant memory entries found."

                response = "Found in collective memory:\n\n"
                for i, entry in enumerate(results, 1):
                    response += f"{i}. {entry['fact']}\n"
                    response += f"   Relevance: {entry['relevance']}\n\n"

                return response

            # Add more commands as needed

        return None

    async def _process_existing_ticket(
        self, user_id: str, user_text: str, ticket: Ticket, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process a message for an existing ticket."""
        # Get assigned agent or re-route if needed
        agent_name = ticket.assigned_to

        # If no valid assignment, route to appropriate agent
        if not agent_name or agent_name not in self.agent_service.get_all_ai_agents():
            agent_name = await self.routing_service.route_query(user_text)
            # Update ticket with new assignment
            self.ticket_service.update_ticket_status(
                ticket.id,
                TicketStatus.ACTIVE,
                assigned_to=agent_name
            )

        # Update ticket status
        self.ticket_service.update_ticket_status(
            ticket.id,
            TicketStatus.ACTIVE
        )

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Generate response with streaming
        full_response = ""
        handoff_info = None

        async for chunk in self.agent_service.generate_response(
            agent_name,
            user_id,
            user_text,
            memory_context,
            temperature=0.7
        ):
            yield chunk
            full_response += chunk

            # Detect handoff requests (simplified - you'd need to implement this detection)
            if "HANDOFF:" in chunk and not handoff_info:
                handoff_pattern = r"HANDOFF:\s*([A-Za-z0-9_]+)\s*REASON:\s*(.+)"
                match = re.search(handoff_pattern, full_response)
                if match:
                    target_agent = match.group(1)
                    reason = match.group(2)
                    handoff_info = {"target": target_agent, "reason": reason}

        # Store conversation in memory if available
        if self.memory_provider:
            await self.memory_provider.store(
                user_id,
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": self._truncate(
                        full_response, 2500)}
                ]
            )

        # Process handoff if detected
        if handoff_info:
            try:
                await self.handoff_service.process_handoff(
                    ticket.id,
                    agent_name,
                    handoff_info["target"],
                    handoff_info["reason"]
                )
            except ValueError as e:
                # If handoff fails, just continue with current agent
                print(f"Handoff failed: {e}")

        # Check if ticket can be considered resolved
        if not handoff_info:
            resolution = await self._check_ticket_resolution(user_id, full_response, user_text)

            if resolution.status == "resolved" and resolution.confidence >= 0.7:
                self.ticket_service.mark_ticket_resolved(
                    ticket.id,
                    {
                        "confidence": resolution.confidence,
                        "reasoning": resolution.reasoning
                    }
                )

                # Create NPS survey
                self.nps_service.create_survey(user_id, ticket.id, agent_name)

        # Extract and store insights in background
        if full_response:
            asyncio.create_task(
                self._extract_and_store_insights(
                    user_id,
                    {"message": user_text, "response": full_response}
                )
            )

    async def _process_new_ticket(
        self, user_id: str, user_text: str, complexity: Dict[str, Any], timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process a message creating a new ticket."""
        # Route query to appropriate agent
        agent_name = await self.routing_service.route_query(user_text)

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Create ticket
        ticket = await self.ticket_service.get_or_create_ticket(
            user_id,
            user_text,
            complexity
        )

        # Update with routing decision
        self.ticket_service.update_ticket_status(
            ticket.id,
            TicketStatus.ACTIVE,
            assigned_to=agent_name
        )

        # Generate response with streaming
        full_response = ""
        handoff_info = None

        async for chunk in self.agent_service.generate_response(
            agent_name,
            user_id,
            user_text,
            memory_context,
            temperature=0.7
        ):
            yield chunk
            full_response += chunk

            # Detect handoff requests (simplified)
            if "HANDOFF:" in chunk and not handoff_info:
                handoff_pattern = r"HANDOFF:\s*([A-Za-z0-9_]+)\s*REASON:\s*(.+)"
                match = re.search(handoff_pattern, full_response)
                if match:
                    target_agent = match.group(1)
                    reason = match.group(2)
                    handoff_info = {"target": target_agent, "reason": reason}

        # Store conversation in memory if available
        if self.memory_provider:
            await self.memory_provider.store(
                user_id,
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": self._truncate(
                        full_response, 2500)}
                ]
            )

        # Process handoff if detected
        if handoff_info:
            try:
                await self.handoff_service.process_handoff(
                    ticket.id,
                    agent_name,
                    handoff_info["target"],
                    handoff_info["reason"]
                )
            except ValueError as e:
                print(f"Handoff failed: {e}")

                # Process handoff if detected
                if handoff_info:
                    try:
                        await self.handoff_service.process_handoff(
                            ticket.id,
                            agent_name,
                            handoff_info["target"],
                            handoff_info["reason"]
                        )
                    except ValueError as e:
                        print(f"Handoff failed: {e}")

                # Check if ticket can be considered resolved
                if not handoff_info:
                    resolution = await self._check_ticket_resolution(user_id, full_response, user_text)

                    if resolution.status == "resolved" and resolution.confidence >= 0.7:
                        self.ticket_service.mark_ticket_resolved(
                            ticket.id,
                            {
                                "confidence": resolution.confidence,
                                "reasoning": resolution.reasoning
                            }
                        )

                        # Create NPS survey
                        self.nps_service.create_survey(
                            user_id, ticket.id, agent_name)

                # Extract and store insights in background
                if full_response:
                    asyncio.create_task(
                        self._extract_and_store_insights(
                            user_id,
                            {"message": user_text, "response": full_response}
                        )
                    )

    async def _process_human_agent_message(self, user_id: str, user_text: str) -> AsyncGenerator[str, None]:
        """Process messages from human agents."""
        # Parse for target agent specification if available
        target_agent = None
        message = user_text

        # Check if message starts with @agent_name to target specific agent
        if user_text.startswith('@'):
            parts = user_text.split(" ", 1)
            potential_target = parts[0][1:]  # Remove the @ symbol
            if potential_target in self.agent_service.get_all_ai_agents():
                target_agent = potential_target
                message = parts[1] if len(parts) > 1 else ""

        # Handle specific commands
        if message.lower() == "!agents":
            yield self._get_agent_directory()
            return

        if message.lower().startswith("!status"):
            yield await self._get_system_status()
            return

        # If no target and no command, provide help
        if not target_agent and not message.strip().startswith("!"):
            yield "Please specify a target AI agent with @agent_name or use a command. Available commands:\n"
            yield "- !agents: List available agents\n"
            yield "- !status: Show system status"
            return

        # Process with target agent
        if target_agent:
            memory_context = ""
            if self.memory_provider:
                memory_context = await self.memory_provider.retrieve(target_agent)

            async for chunk in self.agent_service.generate_response(
                target_agent,
                user_id,
                message,
                memory_context,
                temperature=0.7
            ):
                yield chunk

    def _get_agent_directory(self) -> str:
        """Get formatted list of all registered agents."""
        ai_agents = self.agent_service.get_all_ai_agents()
        human_agents = self.agent_service.get_all_human_agents()
        specializations = self.agent_service.get_specializations()

        result = "# Registered Agents\n\n"

        # AI Agents
        result += "## AI Agents\n\n"
        for name in ai_agents:
            result += f"- **{name}**: {specializations.get(name, 'No specialization')}\n"

        # Human Agents
        if human_agents:
            result += "\n## Human Agents\n\n"
            for agent_id, agent in human_agents.items():
                status = agent.get("availability_status", "unknown")
                name = agent.get("name", agent_id)
                status_emoji = "🟢" if status == "available" else "🔴"
                result += f"- {status_emoji} **{name}**: {agent.get('specialization', 'No specialization')}\n"

        return result

    async def _get_system_status(self) -> str:
        """Get system status summary."""
        # Get ticket metrics
        open_tickets = self.ticket_service.ticket_repository.count(
            {"status": {"$ne": TicketStatus.RESOLVED}})
        resolved_today = self.ticket_service.ticket_repository.count({
            "status": TicketStatus.RESOLVED,
            "resolved_at": {"$gte": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)}
        })

        # Get memory metrics
        memory_count = 0
        try:
            memory_count = self.memory_service.memory_repository.db.count_documents(
                "collective_memory", {})
        except Exception:
            pass

        result = "# System Status\n\n"
        result += f"- Open tickets: {open_tickets}\n"
        result += f"- Resolved in last 24h: {resolved_today}\n"
        result += f"- Collective memory entries: {memory_count}\n"

        return result

    async def _check_ticket_resolution(self, user_id: str, response: str, query: str) -> TicketResolution:
        """Determine if a ticket can be considered resolved based on the response."""
        # Get first AI agent for analysis
        first_agent = next(
            iter(self.agent_service.get_all_ai_agents().keys()))

        prompt = f"""
        Analyze this conversation and determine if the user query has been fully resolved.
        
        USER QUERY: {query}
        
        ASSISTANT RESPONSE: {response}
        
        Determine if this query is:
        1. "resolved" - The user's question/request has been fully addressed
        2. "needs_followup" - The assistant couldn't fully address the issue or more information is needed
        3. "cannot_determine" - Cannot tell if the issue is resolved
        
        Return a JSON object with:
        - "status": One of the above values
        - "confidence": A score from 0.0 to 1.0 indicating confidence in this assessment
        - "reasoning": Brief explanation for your decision
        - "suggested_actions": Array of recommended next steps (if any)
        """

        # Generate resolution assessment
        resolution_text = ""
        async for chunk in self.agent_service.generate_response(
            first_agent,
            "resolution_checker",
            prompt,
            "",  # No memory context needed
            stream=False,
            temperature=0.2,
            response_format={"type": "json_object"}
        ):
            resolution_text += chunk

        try:
            data = json.loads(resolution_text)
            return TicketResolution(**data)
        except Exception as e:
            print(f"Error parsing resolution decision: {e}")
            return TicketResolution(
                status="cannot_determine",
                confidence=0.2,
                reasoning="Failed to analyze resolution status"
            )

    async def _assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess the complexity of a task using standardized metrics."""
        # Get first AI agent for analysis
        first_agent = next(
            iter(self.agent_service.get_all_ai_agents().keys()))

        prompt = f"""
        Analyze this task and provide standardized complexity metrics:
        
        TASK: {query}
        
        Assess on these dimensions:
        1. T-shirt size (XS, S, M, L, XL, XXL)
        2. Story points (1, 2, 3, 5, 8, 13, 21)
        3. Estimated resolution time in minutes/hours
        4. Technical complexity (1-10)
        5. Domain knowledge required (1-10)
        """

        try:
            response_text = ""
            async for chunk in self.agent_service.generate_response(
                first_agent,
                "complexity_assessor",
                prompt,
                "",  # No memory context needed
                stream=False,
                temperature=0.2,
                response_format={"type": "json_object"}
            ):
                response_text += chunk

            complexity_data = json.loads(response_text)
            return complexity_data
        except Exception as e:
            print(f"Error assessing complexity: {e}")
            return {
                "t_shirt_size": "M",
                "story_points": 3,
                "estimated_minutes": 30,
                "technical_complexity": 5,
                "domain_knowledge": 5
            }

    async def _extract_and_store_insights(self, user_id: str, conversation: Dict[str, str]) -> None:
        """Extract insights from conversation and store in collective memory."""
        try:
            # Extract insights
            insights = await self.memory_service.extract_insights(user_id, conversation)

            # Store them if any found
            if insights:
                await self.memory_service.store_insights(user_id, insights)

            return len(insights)
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return 0

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind('.')
        if last_period > limit * 0.8:  # Only use period if it's reasonably close to the end
            return truncated[:last_period + 1]

        return truncated + "..."


#############################################
# FACTORY AND DEPENDENCY INJECTION
#############################################

class SolanaAgentFactory:
    """Factory for creating and wiring components of the Solana Agent system."""

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> QueryProcessor:
        """Create the agent system from configuration."""
        # Create adapters
        db_adapter = MongoDBAdapter(
            connection_string=config["mongo"]["connection_string"],
            database_name=config["mongo"]["database"]
        )

        llm_adapter = OpenAIAdapter(
            api_key=config["openai"]["api_key"],
            model=config.get("openai", {}).get("default_model", "gpt-4o")
        )

        # Create memory providers if configured
        memory_provider = None
        if "zep" in config:
            memory_provider = ZepMemoryAdapter(
                api_key=config["zep"].get("api_key"),
                base_url=config["zep"].get("base_url")
            )

        # Create vector store provider if configured
        vector_provider = None
        if "pinecone" in config:
            vector_provider = PineconeAdapter(
                api_key=config["pinecone"]["api_key"],
                index_name=config["pinecone"]["index"],
                embedding_model=config["pinecone"].get(
                    "embedding_model", "text-embedding-3-small")
            )

        # Create repositories
        ticket_repo = MongoTicketRepository(db_adapter)
        handoff_repo = MongoHandoffRepository(db_adapter)
        nps_repo = MongoNPSSurveyRepository(db_adapter)
        memory_repo = MongoMemoryRepository(db_adapter, vector_provider)

        # Create services
        agent_service = AgentService(llm_adapter)
        routing_service = RoutingService(
            llm_adapter,
            agent_service,
            router_model=config.get("router_model", "gpt-4o-mini")
        )
        ticket_service = TicketService(ticket_repo)
        handoff_service = HandoffService(
            handoff_repo, ticket_repo, agent_service)
        memory_service = MemoryService(memory_repo, llm_adapter)
        nps_service = NPSService(nps_repo, ticket_repo)

        # Create critic service if enabled
        critic_service = None
        if config.get("enable_critic", True):
            critic_service = CriticService(llm_adapter)

        # Create main processor
        query_processor = QueryProcessor(
            agent_service=agent_service,
            routing_service=routing_service,
            ticket_service=ticket_service,
            handoff_service=handoff_service,
            memory_service=memory_service,
            nps_service=nps_service,
            critic_service=critic_service,
            memory_provider=memory_provider,
            enable_critic=config.get("enable_critic", True),
            router_model=config.get("router_model", "gpt-4o-mini")
        )

        # Register predefined agents if any
        for agent_config in config.get("agents", []):
            agent_service.register_ai_agent(
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                specialization=agent_config["specialization"],
                model=agent_config.get("model", "gpt-4o")
            )

        return query_processor


#############################################
# SIMPLIFIED CLIENT INTERFACE
#############################################

class SolanaAgent:
    """Simplified client interface for interacting with the agent system."""

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """Initialize the agent system from config file or dictionary."""
        if not config and not config_path:
            raise ValueError("Either config or config_path must be provided")

        if config_path:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Assume it's a Python file
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        self.processor = SolanaAgentFactory.create_from_config(config)

    async def process(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """Process a user message and return the response stream."""
        async for chunk in self.processor.process(user_id, message):
            yield chunk

    def register_agent(self, name: str, instructions: str, specialization: str, model: str = "gpt-4o") -> None:
        """Register a new AI agent."""
        self.processor.agent_service.register_ai_agent(
            name=name,
            instructions=instructions,
            specialization=specialization,
            model=model
        )

    def register_human_agent(self, agent_id: str, name: str, specialization: str, notification_handler=None) -> None:
        """Register a human agent."""
        self.processor.agent_service.register_human_agent(
            agent_id=agent_id,
            name=name,
            specialization=specialization,
            notification_handler=notification_handler
        )                    #
