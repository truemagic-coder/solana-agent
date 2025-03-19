"""
Tests for notification adapter implementations.

This module contains unit tests for the NotificationProvider implementations.
"""
import pytest
import datetime
from unittest.mock import MagicMock

from solana_agent.adapters.notification_adapter import NullNotificationProvider


@pytest.fixture
def null_provider():
    """Create a null notification provider for testing."""
    return NullNotificationProvider()


class TestNullNotificationProvider:
    """Tests for the NullNotificationProvider."""

    def test_init(self):
        """Test initialization of NullNotificationProvider."""
        provider = NullNotificationProvider()
        assert provider._scheduled_notifications == {}

    @pytest.mark.asyncio
    async def test_send_notification(self, null_provider):
        """Test sending a notification."""
        result = await null_provider.send_notification(
            user_id="test_user",
            message="Test message",
            channel="test_channel"
        )

        # Should always return True without doing anything
        assert result is True

    @pytest.mark.asyncio
    async def test_send_notification_with_metadata(self, null_provider):
        """Test sending a notification with metadata."""
        metadata = {"priority": "high", "category": "alert"}

        result = await null_provider.send_notification(
            user_id="test_user",
            message="Test message with metadata",
            channel="test_channel",
            metadata=metadata
        )

        # Should always return True without doing anything
        assert result is True

    @pytest.mark.asyncio
    async def test_send_scheduled_notification(self, null_provider):
        """Test scheduling a notification."""
        schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)

        notification_id = await null_provider.send_scheduled_notification(
            user_id="test_user",
            message="Scheduled test message",
            channel="test_channel",
            schedule_time=schedule_time
        )

        # Should generate a notification ID with expected format
        assert notification_id.startswith("null_notification_")
        # Should contain a timestamp
        assert "_" in notification_id
        timestamp_part = notification_id.split("_")[-1]
        # Verify it's a valid float (timestamp)
        assert float(timestamp_part)

    @pytest.mark.asyncio
    async def test_send_scheduled_notification_with_metadata(self, null_provider):
        """Test scheduling a notification with metadata."""
        schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        metadata = {"priority": "high", "category": "reminder"}

        notification_id = await null_provider.send_scheduled_notification(
            user_id="test_user",
            message="Scheduled test message with metadata",
            channel="test_channel",
            schedule_time=schedule_time,
            metadata=metadata
        )

        # Should generate a notification ID with expected format
        assert notification_id.startswith("null_notification_")

    @pytest.mark.asyncio
    async def test_cancel_scheduled_notification(self, null_provider):
        """Test canceling a scheduled notification."""
        # First schedule a notification
        schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        notification_id = await null_provider.send_scheduled_notification(
            user_id="test_user",
            message="Will be canceled",
            channel="test_channel",
            schedule_time=schedule_time
        )

        # Now cancel it
        result = await null_provider.cancel_scheduled_notification(notification_id)

        # Should always return True
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_notification(self, null_provider):
        """Test canceling a notification that doesn't exist."""
        result = await null_provider.cancel_scheduled_notification("nonexistent_id")

        # Should still return True since this is a null implementation
        assert result is True

    @pytest.mark.asyncio
    async def test_different_notification_ids(self, null_provider):
        """Test that different notification IDs are generated."""
        schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)

        notification_id1 = await null_provider.send_scheduled_notification(
            user_id="test_user",
            message="First message",
            channel="test_channel",
            schedule_time=schedule_time
        )

        notification_id2 = await null_provider.send_scheduled_notification(
            user_id="test_user",
            message="Second message",
            channel="test_channel",
            schedule_time=schedule_time
        )

        # IDs should be different
        assert notification_id1 != notification_id2
