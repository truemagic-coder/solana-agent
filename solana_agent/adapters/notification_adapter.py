"""
Notification adapters for the Solana Agent system.

These adapters implement notification services for sending alerts to users and agents.
"""
import datetime
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Optional, Any, List

from solana_agent.interfaces.providers import NotificationProvider


class EmailNotificationAdapter(NotificationProvider):
    """Email-based notification provider."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.username = username or sender_email
        self.password = password
        self.use_tls = use_tls
        self._scheduled_notifications = {}  # id -> notification details

    async def send_notification(self, user_id: str, message: str, channel: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification to a user via email."""
        if channel != "email" or "@" not in user_id:
            return False

        try:
            # Create message
            msg = MIMEText(message)
            msg["Subject"] = metadata.get(
                "subject", "Notification from Solana Agent") if metadata else "Notification from Solana Agent"
            msg["From"] = self.sender_email
            msg["To"] = user_id

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()

                if self.password:
                    server.login(self.username, self.password)

                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Error sending email notification: {e}")
            return False

    async def send_scheduled_notification(self, user_id: str, message: str, channel: str, schedule_time: datetime, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a notification to be sent later."""
        notification_id = f"notification_{len(self._scheduled_notifications) + 1}"

        self._scheduled_notifications[notification_id] = {
            "user_id": user_id,
            "message": message,
            "channel": channel,
            "schedule_time": schedule_time,
            "metadata": metadata or {}
        }

        # In a real implementation, you would use a task scheduler like APScheduler
        # For now, we just store it in memory

        return notification_id

    async def cancel_scheduled_notification(self, schedule_id: str) -> bool:
        """Cancel a scheduled notification."""
        if schedule_id in self._scheduled_notifications:
            del self._scheduled_notifications[schedule_id]
            return True
        return False
