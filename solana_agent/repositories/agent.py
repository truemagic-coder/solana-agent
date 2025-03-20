"""
MongoDB implementation of the agent repository.
"""
from typing import List, Optional, Any

from solana_agent.domains.agents import AIAgent
from solana_agent.interfaces.repositories.agent import AgentRepository


class MongoAgentRepository(AgentRepository):
    """MongoDB implementation of the AgentRepository interface."""

    def __init__(self, db_adapter):
        """Initialize the repository with a database adapter."""
        self.db = db_adapter
        self.ai_agents_collection = "agents"

        # Ensure collections exist
        self.db.create_collection(self.ai_agents_collection)

        # Create indexes
        self.db.create_index(self.ai_agents_collection,
                             [("name", 1)], unique=True)

    def get_ai_agent_by_name(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name.

        Args:
            name: The name of the AI agent

        Returns:
            The AI agent or None if not found
        """
        # Query the AI agents collection for a document with matching name
        doc = self.db.find_one(self.ai_agents_collection, {"name": name})

        # If no document found, return None
        if not doc:
            return None

        # Convert the document to an AIAgent domain model
        try:
            return AIAgent.model_validate(doc)
        except Exception as e:
            print(f"Error parsing AI agent with name {name}: {e}")
            return None

    def get_ai_agent(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        doc = self.db.find_one(self.ai_agents_collection, {"name": name})
        if not doc:
            return None

        return AIAgent.model_validate(doc)

    def get_all_ai_agents(self) -> List[AIAgent]:
        """Get all AI agents in the system.

        Returns:
            List of all AI agents
        """
        # Query all documents from the AI agents collection
        docs = self.db.find(self.ai_agents_collection, {})

        # Convert each document to an AIAgent domain model
        ai_agents = []
        for doc in docs:
            try:
                agent = AIAgent.model_validate(doc)
                ai_agents.append(agent)
            except Exception as e:
                # Log the error but continue processing other agents
                print(f"Error parsing AI agent from database: {e}")

        return ai_agents

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
