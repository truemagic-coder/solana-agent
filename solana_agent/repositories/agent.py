"""
MongoDB implementation of the agent repository.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from solana_agent.domain.agents import AIAgent, HumanAgent, AgentPerformance
from solana_agent.domain.enums import AgentType
from solana_agent.interfaces.repositories import AgentRepository


class MongoAgentRepository(AgentRepository):
    """MongoDB implementation of the AgentRepository interface."""

    def __init__(self, db_adapter):
        """Initialize the repository with a database adapter."""
        self.db = db_adapter
        self.ai_agents_collection = "ai_agents"
        self.human_agents_collection = "human_agents"
        self.performance_collection = "agent_performance"

        # Ensure collections exist
        self.db.create_collection(self.ai_agents_collection)
        self.db.create_collection(self.human_agents_collection)
        self.db.create_collection(self.performance_collection)

        # Create indexes
        self.db.create_index(self.ai_agents_collection,
                             [("name", 1)], unique=True)
        self.db.create_index(self.human_agents_collection,
                             [("id", 1)], unique=True)
        self.db.create_index(self.human_agents_collection,
                             [("email", 1)], unique=True)
        self.db.create_index(self.performance_collection, [("agent_id", 1)])
        self.db.create_index(self.performance_collection,
                             [("period_start", 1)])

    def get_ai_agent(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        doc = self.db.find_one(self.ai_agents_collection, {"name": name})
        if not doc:
            return None

        return AIAgent.model_validate(doc)

    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
        """Get all AI agents."""
        docs = self.db.find(self.ai_agents_collection, {})

        return {
            doc["name"]: AIAgent.model_validate(doc)
            for doc in docs
        }

    def save_ai_agent(self, agent: AIAgent) -> bool:
        """Save an AI agent."""
        doc = agent.model_dump()

        # Check if agent already exists
        existing = self.db.find_one(
            self.ai_agents_collection, {"name": agent.name})

        if existing:
            # Update existing agent
            return self.db.update_one(
                self.ai_agents_collection,
                {"name": agent.name},
                {"$set": doc}
            )
        else:
            # Create new agent
            self.db.insert_one(self.ai_agents_collection, doc)
            return True

    def delete_ai_agent(self, name: str) -> bool:
        """Delete an AI agent."""
        return self.db.delete_one(self.ai_agents_collection, {"name": name})

    def get_human_agent(self, agent_id: str) -> Optional[HumanAgent]:
        """Get a human agent by ID."""
        doc = self.db.find_one(self.human_agents_collection, {"id": agent_id})
        if not doc:
            return None

        return HumanAgent.model_validate(doc)

    def get_human_agents_by_specialization(self, specialization: str) -> List[HumanAgent]:
        """Get human agents with a specific specialization."""
        docs = self.db.find(
            self.human_agents_collection,
            {"specializations": specialization, "is_active": True}
        )

        return [HumanAgent.model_validate(doc) for doc in docs]

    def get_available_human_agents(self) -> List[HumanAgent]:
        """Get currently available human agents."""
        docs = self.db.find(
            self.human_agents_collection,
            {"is_active": True}
        )

        # Filter to those actually available at current time
        agents = []
        for doc in docs:
            agent = HumanAgent.model_validate(doc)
            if agent.is_available_now():
                agents.append(agent)

        return agents

    def save_human_agent(self, agent: HumanAgent) -> bool:
        """Save a human agent."""
        doc = agent.model_dump()

        # Check if agent already exists
        existing = self.db.find_one(
            self.human_agents_collection, {"id": agent.id})

        if existing:
            # Update existing agent
            return self.db.update_one(
                self.human_agents_collection,
                {"id": agent.id},
                {"$set": doc}
            )
        else:
            # Create new agent
            self.db.insert_one(self.human_agents_collection, doc)
            return True

    def save_agent_performance(self, performance: AgentPerformance) -> bool:
        """Save agent performance metrics."""
        doc = performance.model_dump()

        # Convert datetime to string for MongoDB
        if isinstance(doc["period_start"], datetime):
            doc["period_start"] = doc["period_start"].isoformat()

        if isinstance(doc["period_end"], datetime):
            doc["period_end"] = doc["period_end"].isoformat()

        # Check if performance record already exists
        existing = self.db.find_one(
            self.performance_collection,
            {
                "agent_id": performance.agent_id,
                "period_start": doc["period_start"]
            }
        )

        if existing:
            # Update existing record
            return self.db.update_one(
                self.performance_collection,
                {"_id": existing["_id"]},
                {"$set": doc}
            )
        else:
            # Create new record
            self.db.insert_one(self.performance_collection, doc)
            return True

    def get_agent_performance(
        self, agent_id: str, period_start: datetime, period_end: datetime
    ) -> Optional[AgentPerformance]:
        """Get performance metrics for an agent within a time period."""
        period_start_str = period_start.isoformat()
        period_end_str = period_end.isoformat()

        doc = self.db.find_one(
            self.performance_collection,
            {
                "agent_id": agent_id,
                "period_start": period_start_str,
                "period_end": period_end_str
            }
        )

        if not doc:
            return None

        # Convert string dates back to datetime
        if isinstance(doc["period_start"], str):
            doc["period_start"] = datetime.fromisoformat(doc["period_start"])

        if isinstance(doc["period_end"], str):
            doc["period_end"] = datetime.fromisoformat(doc["period_end"])

        return AgentPerformance.model_validate(doc)
