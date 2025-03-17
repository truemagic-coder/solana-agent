"""
MongoDB implementation of the feedback repository.

This repository handles storing and retrieving user feedback and NPS ratings.
"""
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Any

from solana_agent.interfaces.repositories import FeedbackRepository
from solana_agent.interfaces.providers import DataStorageProvider
from solana_agent.domain.feedback import UserFeedback, FeedbackType, NPSRating


class MongoFeedbackRepository(FeedbackRepository):
    """MongoDB implementation of FeedbackRepository."""

    def __init__(self, db_adapter: DataStorageProvider):
        """Initialize the feedback repository.

        Args:
            db_adapter: MongoDB adapter
        """
        self.db = db_adapter
        self.collection = "feedback"

        # Ensure collections exist
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("user_id", 1)])
        self.db.create_index(self.collection, [("timestamp", -1)])
        self.db.create_index(self.collection, [("type", 1)])

    def store_feedback(self, feedback: UserFeedback) -> str:
        """Store user feedback.

        Args:
            feedback: Feedback object to store

        Returns:
            Feedback ID
        """
        # Generate ID if not present
        if not feedback.id:
            feedback.id = str(uuid.uuid4())

        # Convert to dictionary
        feedback_dict = feedback.model_dump()

        # Store in MongoDB
        self.db.insert_one(self.collection, feedback_dict)

        return feedback.id

    def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback for a user.

        Args:
            user_id: User ID

        Returns:
            List of feedback items
        """
        # Query MongoDB
        feedback_docs = self.db.find(
            self.collection,
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )

        # Convert to UserFeedback objects
        feedback_items = []
        for doc in feedback_docs:
            try:
                # Handle NPSRating nested object if present
                if doc.get("type") == FeedbackType.NPS and isinstance(doc.get("nps_rating"), dict):
                    nps_data = doc.get("nps_rating")
                    doc["nps_rating"] = NPSRating(**nps_data)

                feedback = UserFeedback(**doc)
                feedback_items.append(feedback)
            except Exception as e:
                print(f"Error parsing feedback: {e}")

        return feedback_items

    def get_average_nps(self, days: int = 30) -> float:
        """Calculate average NPS score for a time period.

        Args:
            days: Number of days to include

        Returns:
            Average NPS score
        """
        # Calculate date cutoff
        cutoff_date = datetime.now() - timedelta(days=days)

        # Query MongoDB for NPS ratings
        pipeline = [
            {
                "$match": {
                    "type": FeedbackType.NPS,
                    "timestamp": {"$gte": cutoff_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$nps_rating.score"},
                    "count": {"$sum": 1}
                }
            }
        ]

        result = self.db.aggregate(self.collection, pipeline)

        if result and len(result) > 0:
            return result[0].get("avg_score", 0.0)

        return 0.0

    def get_nps_distribution(self, days: int = 30) -> Dict[int, int]:
        """Get distribution of NPS scores for a time period.

        Args:
            days: Number of days to include

        Returns:
            Dictionary mapping scores to counts
        """
        # Calculate date cutoff
        cutoff_date = datetime.now() - timedelta(days=days)

        # Query MongoDB for NPS ratings
        pipeline = [
            {
                "$match": {
                    "type": FeedbackType.NPS,
                    "timestamp": {"$gte": cutoff_date}
                }
            },
            {
                "$group": {
                    "_id": "$nps_rating.score",
                    "count": {"$sum": 1}
                }
            }
        ]

        result = self.db.aggregate(self.collection, pipeline)

        # Convert to dictionary
        distribution = {}
        for item in result:
            score = item.get("_id")
            if score is not None:
                distribution[int(score)] = item.get("count", 0)

        # Ensure all scores 0-10 are represented
        for score in range(11):
            if score not in distribution:
                distribution[score] = 0

        return distribution
