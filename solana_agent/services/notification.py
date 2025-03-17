"""
Notification service implementation.

This service manages sending notifications through various channels
and scheduling notifications for future delivery.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from solana_agent.interfaces.services import NotificationService as NotificationServiceInterface
from solana_agent.interfaces.providers import NotificationProvider
from solana_agent.domain.notifications import NotificationTemplate


class NotificationService(NotificationServiceInterface):
    """Service for managing and sending notifications."""

    def __init__(self, notification_provider: NotificationProvider):
        """Initialize the notification service.

        Args:
            notification_provider: Provider for sending notifications
        """
        self.provider = notification_provider
        self.templates: Dict[str, NotificationTemplate] = {}

    async def send_notification(
        self,
        user_id: str,
        message: str,
        channel: str = "email",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification to a user.

        Args:
            user_id: User ID or email address
            message: Notification message
            channel: Notification channel (email, sms, in_app)
            metadata: Additional metadata

        Returns:
            True if notification was sent successfully
        """
        return await self.provider.send_notification(
            user_id=user_id,
            message=message,
            channel=channel,
            metadata=metadata or {}
        )

    async def send_from_template(
        self,
        user_id: str,
        template_id: str,
        data: Dict[str, Any],
        channel: str = "email"
    ) -> bool:
        """Send a notification using a template.

        Args:
            user_id: User ID or email address
            template_id: Template identifier
            data: Template data for substitution
            channel: Notification channel

        Returns:
            True if notification was sent successfully
        """
        template = self.templates.get(template_id)
        if not template:
            print(f"Template not found: {template_id}")
            return False

        # Process template with data
        try:
            message = template.render(data)
            metadata = {
                "subject": template.format_subject(data),
                "template_id": template_id
            }

            # Send notification
            return await self.send_notification(
                user_id=user_id,
                message=message,
                channel=channel,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error processing template: {e}")
            return False

    def register_template(self, template_id: str, template: NotificationTemplate) -> None:
        """Register a notification template.

        Args:
            template_id: Template identifier
            template: Template object
        """
        self.templates[template_id] = template

    async def schedule_notification(
        self,
        user_id: str,
        message: str,
        scheduled_time: datetime,
        channel: str = "email",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a notification for future delivery.

        Args:
            user_id: User ID or email address
            message: Notification message
            scheduled_time: When to send the notification
            channel: Notification channel
            metadata: Additional metadata

        Returns:
            Scheduled notification ID
        """
        return await self.provider.send_scheduled_notification(
            user_id=user_id,
            message=message,
            channel=channel,
            schedule_time=scheduled_time,
            metadata=metadata or {}
        )

    async def cancel_scheduled_notification(self, notification_id: str) -> bool:
        """Cancel a scheduled notification.

        Args:
            notification_id: Scheduled notification ID

        Returns:
            True if cancellation was successful
        """
        return await self.provider.cancel_scheduled_notification(notification_id)
