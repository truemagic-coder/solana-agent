"""
Feedback domain models.

These models define structures for user feedback and NPS ratings.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of user feedback."""
    NPS = "nps"
    TEXT = "text"
    RATING = "rating"
    ISSUE = "issue"


class NPSRating(BaseModel):
    """Net Promoter Score rating."""
    score: int = Field(..., description="NPS score (0-10)", ge=0, le=10)
    reason: Optional[str] = Field(
        None, description="Reason for the score provided")

    @property
    def category(self) -> str:
        """Get the NPS category based on score."""
        if self.score >= 9:
            return "promoter"
        elif self.score >= 7:
            return "passive"
        else:
            return "detractor"


class UserFeedback(BaseModel):
    """User feedback model."""
    id: str = Field("", description="Unique identifier")
    user_id: str = Field(...,
                         description="ID of the user who provided the feedback")
    type: FeedbackType = Field(..., description="Type of feedback")
    ticket_id: Optional[str] = Field(
        None, description="Related ticket ID if applicable")
    text: Optional[str] = Field(None, description="Text feedback content")
    nps_rating: Optional[NPSRating] = Field(
        None, description="NPS rating details")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the feedback was provided")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")
