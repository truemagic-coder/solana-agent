"""
MongoDB implementation of HandoffRepository.

This repository handles storing and retrieving handoff records.
"""
import datetime
from typing import List, Optional, Any

from solana_agent.interfaces import HandoffRepository
from solana_agent.interfaces import DBProvider
from solana_agent.domains import Handoff


class MongoHandoffRepository(HandoffRepository):
    """MongoDB implementation of HandoffRepository."""

    def __init__(self, db_provider: DBProvider):
        """Initialize the handoff repository.

        Args:
            db_provider: Provider for database operations
        """
        self.db = db_provider
        self.collection = "handoffs"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("from_agent", 1)])
        self.db.create_index(self.collection, [("to_agent", 1)])
        self.db.create_index(self.collection, [("timestamp", 1)])

    def record(self, handoff: Handoff) -> str:
        """Record a new handoff.

        Args:
            handoff: Handoff object

        Returns:
            Handoff ID
        """
        return self.db.insert_one(self.collection, handoff.model_dump(mode="json"))

    def find_for_agent(
        self,
        agent_name: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[Handoff]:
        """Find handoffs for an agent.

        Args:
            agent_name: Agent name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of handoff objects
        """
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
        end_date: Optional[datetime.datetime] = None,
    ) -> int:
        """Count handoffs for an agent.

        Args:
            agent_name: Agent name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Number of handoffs
        """
        query = {"from_agent": agent_name}

        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        return self.db.count_documents(self.collection, query)
