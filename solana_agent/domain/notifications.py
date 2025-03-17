"""
Notification domain models.

These models define structures for notifications and templates.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    SLACK = "slack"
    WEBHOOK = "webhook"


class NotificationStatus(str, Enum):
    """Status of a notification."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Notification(BaseModel):
    """Notification model."""
    id: str = Field("", description="Unique identifier")
    user_id: str = Field(..., description="ID of the recipient")
    message: str = Field(..., description="Notification message")
    channel: NotificationChannel = Field(
        NotificationChannel.EMAIL, description="Delivery channel")
    status: NotificationStatus = Field(
        NotificationStatus.PENDING, description="Notification status")
    scheduled_time: Optional[datetime] = Field(
        None, description="When to send the notification")
    sent_time: Optional[datetime] = Field(
        None, description="When the notification was sent")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class NotificationTemplate(BaseModel):
    """Template for notifications."""
    id: str = Field("", description="Unique identifier")
    name: str = Field(..., description="Template name")
    subject_template: str = Field(
        "", description="Subject template for email notifications")
    body_template: str = Field(..., description="Body template")
    default_channel: NotificationChannel = Field(
        NotificationChannel.EMAIL, description="Default channel")

    def render(self, data: Dict[str, Any]) -> str:
        """Render the body template with data.

        Args:
            data: Template data

        Returns:
            Rendered message
        """
        # Simple replacement-based template rendering
        result = self.body_template
        for key, value in data.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def format_subject(self, data: Dict[str, Any]) -> str:
        """Format the subject template with data.

        Args:
            data: Template data

        Returns:
            Formatted subject
        """
        result = self.subject_template
        for key, value in data.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
