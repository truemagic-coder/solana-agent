"""
Notification adapters for the Solana Agent system.

These adapters implement notification services for sending alerts to users and agents.
"""
import datetime
from typing import Dict, Optional, Any

from solana_agent.interfaces import NotificationProvider


class NullNotificationProvider(NotificationProvider):
    """Null implementation of the NotificationProvider interface.

    This provider satisfies the interface but doesn't actually send any notifications.
    It's useful when notifications aren't needed or when running tests.
    """

    def __init__(self):
        """Initialize the null notification provider."""
        self._scheduled_notifications = {}  # Empty dict for compatibility

    async def send_notification(self, user_id: str, message: str, channel: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Pretend to send a notification.

        Args:
            user_id: ID of the user (ignored)
            message: Message content (ignored)
            channel: Channel to use (ignored)
            metadata: Optional metadata (ignored)

        Returns:
            Always returns True
        """
        # Do nothing, just return success
        return True

    async def send_scheduled_notification(self, user_id: str, message: str, channel: str, schedule_time: datetime, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Pretend to schedule a notification.

        Args:
            user_id: ID of the user (ignored)
            message: Message content (ignored)
            channel: Channel to use (ignored)
            schedule_time: Time to send (ignored)
            metadata: Optional metadata (ignored)

        Returns:
            A dummy notification ID
        """
        # Generate a dummy notification ID
        notification_id = f"null_notification_{datetime.datetime.now().timestamp()}"
        return notification_id

    async def cancel_scheduled_notification(self, schedule_id: str) -> bool:
        """Pretend to cancel a scheduled notification.

        Args:
            schedule_id: ID of the notification to cancel (ignored)

        Returns:
            Always returns True
        """
        # Do nothing, just return success
        return True
