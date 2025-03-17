"""
NPS (Net Promoter Score) service implementation.

This service manages user satisfaction tracking, NPS surveys, and feedback collection.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

from solana_agent.interfaces.services import NPSService as NPSServiceInterface
from solana_agent.interfaces.repositories import FeedbackRepository
from solana_agent.domain.feedback import UserFeedback, NPSRating, FeedbackType


class NPSService(NPSServiceInterface):
    """Service for managing NPS and user feedback."""

    def __init__(self, feedback_repository: FeedbackRepository):
        """Initialize the NPS service.

        Args:
            feedback_repository: Repository for storing and retrieving feedback
        """
        self.feedback_repository = feedback_repository

    def store_nps_rating(self, user_id: str, score: int, ticket_id: Optional[str] = None) -> str:
        """Store an NPS rating from a user.

        Args:
            user_id: User ID
            score: NPS score (0-10)
            ticket_id: Optional ticket ID

        Returns:
            Feedback ID
        """
        # Validate score
        if not 0 <= score <= 10:
            raise ValueError("NPS score must be between 0 and 10")

        # Create feedback object with NPSRating
        feedback = UserFeedback(
            user_id=user_id,
            type=FeedbackType.NPS,
            ticket_id=ticket_id,
            nps_rating=NPSRating(score=score),
            timestamp=datetime.now()
        )

        # Store in repository
        return self.feedback_repository.store_feedback(feedback)

    def store_feedback(self, user_id: str, feedback_text: str, ticket_id: Optional[str] = None) -> str:
        """Store textual feedback from a user.

        Args:
            user_id: User ID
            feedback_text: Feedback text
            ticket_id: Optional ticket ID

        Returns:
            Feedback ID
        """
        # Create feedback object
        feedback = UserFeedback(
            user_id=user_id,
            type=FeedbackType.TEXT,
            ticket_id=ticket_id,
            text=feedback_text,
            timestamp=datetime.now()
        )

        # Store in repository
        return self.feedback_repository.store_feedback(feedback)

    def get_user_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """Get feedback history for a user.

        Args:
            user_id: User ID

        Returns:
            List of feedback items
        """
        return self.feedback_repository.get_user_feedback(user_id)

    def get_average_nps(self, days: int = 30) -> float:
        """Calculate average NPS score for a time period.

        Args:
            days: Number of days to include

        Returns:
            Average NPS score
        """
        return self.feedback_repository.get_average_nps(days)

    def get_nps_distribution(self, days: int = 30) -> Dict[int, int]:
        """Get distribution of NPS scores for a time period.

        Args:
            days: Number of days to include

        Returns:
            Dictionary mapping scores to counts
        """
        return self.feedback_repository.get_nps_distribution(days)

    def calculate_nps_score(self, days: int = 30) -> Dict[str, Any]:
        """Calculate full NPS metrics.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with promoters, detractors, passive, and NPS score
        """
        distribution = self.feedback_repository.get_nps_distribution(days)

        # Initialize counters
        promoters = 0
        passives = 0
        detractors = 0
        total = 0

        # Count each category
        for score, count in distribution.items():
            if score >= 9:  # 9-10 are promoters
                promoters += count
            elif score >= 7:  # 7-8 are passives
                passives += count
            else:  # 0-6 are detractors
                detractors += count
            total += count

        # Calculate percentages and NPS
        if total > 0:
            promoter_percent = (promoters / total) * 100
            detractor_percent = (detractors / total) * 100
            nps = promoter_percent - detractor_percent
        else:
            promoter_percent = 0
            detractor_percent = 0
            nps = 0

        return {
            "promoters": promoters,
            "promoter_percent": promoter_percent,
            "passives": passives,
            "passive_percent": (passives / total) * 100 if total > 0 else 0,
            "detractors": detractors,
            "detractor_percent": detractor_percent,
            "total_responses": total,
            "nps_score": nps
        }
