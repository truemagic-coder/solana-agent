"""
Tests for the NotificationService implementation.

This module tests sending notifications, template management,
and scheduling notifications.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pytz

from solana_agent.services.notification import NotificationService
from solana_agent.domains import NotificationTemplate


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_notification_provider():
    """Return a mock notification provider."""
    provider = Mock()

    # Mock the send_notification method
    provider.send_notification = AsyncMock(return_value=True)

    # Mock the send_scheduled_notification method
    provider.send_scheduled_notification = AsyncMock(
        return_value="notification-123")

    # Mock the cancel_scheduled_notification method
    provider.cancel_scheduled_notification = AsyncMock(return_value=True)

    return provider


@pytest.fixture
def notification_service(mock_notification_provider):
    """Return a notification service with mocked dependencies."""
    return NotificationService(notification_provider=mock_notification_provider)


@pytest.fixture
def sample_template():
    """Return a sample notification template."""
    template = NotificationTemplate(
        name="account_status_template",  # Required field that was missing
        # Changed from body to body_template
        body_template="Hello, {{name}}! Your {{service}} account has been {{status}}.",
        # Changed from subject to subject_template
        subject_template="{{service}} Account {{status}}",
    )
    return template


@pytest.fixture
def registered_template_service(notification_service, sample_template):
    """Return a notification service with a registered template."""
    notification_service.register_template("account_status", sample_template)
    return notification_service


# ---------------------
# Initialization Tests
# ---------------------

def test_notification_service_initialization(mock_notification_provider):
    """Test that the notification service initializes properly."""
    service = NotificationService(
        notification_provider=mock_notification_provider)

    assert service.provider == mock_notification_provider
    assert service.templates == {}


# ---------------------
# Send Notification Tests
# ---------------------

@pytest.mark.asyncio
async def test_send_notification_basic(notification_service):
    """Test sending a basic notification."""
    # Act
    result = await notification_service.send_notification(
        user_id="user123",
        message="Test notification",
        channel="email"
    )

    # Assert
    assert result is True
    notification_service.provider.send_notification.assert_called_once_with(
        user_id="user123",
        message="Test notification",
        channel="email",
        metadata={}
    )


@pytest.mark.asyncio
async def test_send_notification_with_metadata(notification_service):
    """Test sending a notification with metadata."""
    # Arrange
    metadata = {"subject": "Important Alert", "priority": "high"}

    # Act
    result = await notification_service.send_notification(
        user_id="user123",
        message="Test notification with metadata",
        channel="sms",
        metadata=metadata
    )

    # Assert
    assert result is True
    notification_service.provider.send_notification.assert_called_once_with(
        user_id="user123",
        message="Test notification with metadata",
        channel="sms",
        metadata=metadata
    )


@pytest.mark.asyncio
async def test_send_notification_failure(notification_service):
    """Test handling a failed notification."""
    # Arrange
    notification_service.provider.send_notification = AsyncMock(
        return_value=False)

    # Act
    result = await notification_service.send_notification(
        user_id="user123",
        message="Test notification",
        channel="email"
    )

    # Assert
    assert result is False
    notification_service.provider.send_notification.assert_called_once()


# ---------------------
# Template Management Tests
# ---------------------

def test_register_template(notification_service, sample_template):
    """Test registering a notification template."""
    # Act
    notification_service.register_template("welcome_email", sample_template)

    # Assert
    assert "welcome_email" in notification_service.templates
    assert notification_service.templates["welcome_email"] == sample_template


def test_register_multiple_templates(notification_service, sample_template):
    """Test registering multiple templates."""
    # Arrange
    second_template = NotificationTemplate(
        name="verification_template",  # Required field
        # Changed from body to body_template
        body_template="Your verification code is {{code}}",
        subject_template="Verification Code"  # Changed from subject to subject_template
    )

    # Act
    notification_service.register_template("welcome_email", sample_template)
    notification_service.register_template("verification", second_template)

    # Assert
    assert len(notification_service.templates) == 2
    assert notification_service.templates["welcome_email"] == sample_template
    assert notification_service.templates["verification"] == second_template


# ---------------------
# Send From Template Tests
# ---------------------

@pytest.mark.asyncio
async def test_send_from_template_success(registered_template_service):
    """Test successfully sending a notification from a template."""
    # Arrange
    template_data = {
        "name": "John",
        "service": "Solana",
        "status": "activated"
    }

    # Act
    result = await registered_template_service.send_from_template(
        user_id="john@example.com",
        template_id="account_status",
        data=template_data,
        channel="email"
    )

    # Assert
    assert result is True
    registered_template_service.provider.send_notification.assert_called_once()
    # Check that the template was rendered correctly
    call_args = registered_template_service.provider.send_notification.call_args[1]
    assert "Hello, John! Your Solana account has been activated." == call_args["message"]
    assert call_args["metadata"]["subject"] == "Solana Account activated"


@pytest.mark.asyncio
async def test_send_from_nonexistent_template(notification_service):
    """Test sending from a template that doesn't exist."""
    # Act
    result = await notification_service.send_from_template(
        user_id="user123",
        template_id="nonexistent",
        data={},
        channel="email"
    )

    # Assert
    assert result is False
    notification_service.provider.send_notification.assert_not_called()

# ---------------------
# Schedule Notification Tests
# ---------------------


@pytest.mark.asyncio
async def test_schedule_notification(notification_service):
    """Test scheduling a notification."""
    # Arrange
    scheduled_time = datetime.now(pytz.UTC) + timedelta(hours=1)

    # Act
    notification_id = await notification_service.schedule_notification(
        user_id="user123",
        message="Scheduled notification test",
        scheduled_time=scheduled_time,
        channel="email"
    )

    # Assert
    assert notification_id == "notification-123"
    notification_service.provider.send_scheduled_notification.assert_called_once_with(
        user_id="user123",
        message="Scheduled notification test",
        channel="email",
        schedule_time=scheduled_time,
        metadata={}
    )


@pytest.mark.asyncio
async def test_schedule_notification_with_metadata(notification_service):
    """Test scheduling a notification with metadata."""
    # Arrange
    scheduled_time = datetime.now(pytz.UTC) + timedelta(hours=1)
    metadata = {"subject": "Reminder", "category": "meeting"}

    # Act
    notification_id = await notification_service.schedule_notification(
        user_id="user123",
        message="Meeting reminder",
        scheduled_time=scheduled_time,
        channel="in_app",
        metadata=metadata
    )

    # Assert
    assert notification_id == "notification-123"
    notification_service.provider.send_scheduled_notification.assert_called_once_with(
        user_id="user123",
        message="Meeting reminder",
        channel="in_app",
        schedule_time=scheduled_time,
        metadata=metadata
    )


# ---------------------
# Cancel Scheduled Notification Tests
# ---------------------

@pytest.mark.asyncio
async def test_cancel_scheduled_notification(notification_service):
    """Test canceling a scheduled notification."""
    # Act
    result = await notification_service.cancel_scheduled_notification("notification-123")

    # Assert
    assert result is True
    notification_service.provider.cancel_scheduled_notification.assert_called_once_with(
        "notification-123"
    )


@pytest.mark.asyncio
async def test_cancel_scheduled_notification_failure(notification_service):
    """Test failure when canceling a scheduled notification."""
    # Arrange
    notification_service.provider.cancel_scheduled_notification = AsyncMock(
        return_value=False)

    # Act
    result = await notification_service.cancel_scheduled_notification("invalid-id")

    # Assert
    assert result is False
    notification_service.provider.cancel_scheduled_notification.assert_called_once_with(
        "invalid-id"
    )


# ---------------------
# Integration Tests
# ---------------------

@pytest.mark.asyncio
async def test_template_and_schedule_workflow(notification_service, sample_template):
    """Test the workflow of registering a template and scheduling a notification."""
    # Arrange
    notification_service.register_template("reminder", sample_template)
    scheduled_time = datetime.now(pytz.UTC) + timedelta(hours=2)
    template_data = {
        "name": "Alice",
        "service": "Wallet",
        "status": "updated"
    }

    # Act - First send from template
    template_result = await notification_service.send_from_template(
        user_id="alice@example.com",
        template_id="reminder",
        data=template_data
    )

    # Then schedule a notification
    notification_id = await notification_service.schedule_notification(
        user_id="alice@example.com",
        message="Follow-up reminder",
        scheduled_time=scheduled_time
    )

    # Assert
    assert template_result is True
    assert notification_id == "notification-123"
    assert notification_service.provider.send_notification.call_count == 1
    assert notification_service.provider.send_scheduled_notification.call_count == 1
