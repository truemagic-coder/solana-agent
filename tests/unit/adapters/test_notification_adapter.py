"""
Tests for notification adapter implementations.

This module contains unit tests for EmailNotificationAdapter.
"""
import pytest
import datetime
from unittest.mock import patch, MagicMock, call
from email.mime.text import MIMEText

from solana_agent.adapters.notification_adapter import EmailNotificationAdapter


@pytest.fixture
def email_adapter():
    """Create an email notification adapter for testing."""
    return EmailNotificationAdapter(
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender_email="sender@example.com",
        username="username",
        password="password",
        use_tls=True
    )


@pytest.mark.asyncio
async def test_init():
    """Test EmailNotificationAdapter initialization."""
    # Test with all parameters
    adapter = EmailNotificationAdapter(
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender_email="sender@example.com",
        username="username",
        password="password",
        use_tls=True
    )

    assert adapter.smtp_server == "smtp.example.com"
    assert adapter.smtp_port == 587
    assert adapter.sender_email == "sender@example.com"
    assert adapter.username == "username"
    assert adapter.password == "password"
    assert adapter.use_tls is True

    # Test with default username (uses sender_email)
    adapter = EmailNotificationAdapter(
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender_email="sender@example.com",
        password="password"
    )

    assert adapter.username == "sender@example.com"
    assert adapter._scheduled_notifications == {}


@pytest.mark.asyncio
async def test_send_notification_success(email_adapter):
    """Test sending a notification successfully."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test notification message"
    channel = "email"
    metadata = {"subject": "Test Subject"}

    # Mock SMTP server
    mock_server = MagicMock()

    # Mock context manager returned by SMTP constructor
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_server

    # Patch SMTP constructor to return our mock
    with patch('smtplib.SMTP', return_value=mock_context) as mock_smtp:
        # Call the method
        result = await email_adapter.send_notification(user_id, message, channel, metadata)

        # Verify the result
        assert result is True

        # Verify SMTP was initialized correctly
        mock_smtp.assert_called_once_with("smtp.example.com", 587)

        # Verify TLS was started
        mock_server.starttls.assert_called_once()

        # Verify login was called
        mock_server.login.assert_called_once_with("username", "password")

        # Verify message was sent
        mock_server.send_message.assert_called_once()

        # Get the message that was sent
        sent_message = mock_server.send_message.call_args[0][0]

        # Verify message contents
        assert sent_message["From"] == "sender@example.com"
        assert sent_message["To"] == "user@example.com"
        assert sent_message["Subject"] == "Test Subject"
        assert sent_message.get_payload() == "Test notification message"


@pytest.mark.asyncio
async def test_send_notification_wrong_channel(email_adapter):
    """Test sending a notification with wrong channel."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test notification message"
    channel = "sms"  # Not email

    # Mock SMTP server to ensure it's not called
    with patch('smtplib.SMTP') as mock_smtp:
        # Call the method
        result = await email_adapter.send_notification(user_id, message, channel)

        # Verify the result
        assert result is False

        # Verify SMTP was not initialized
        mock_smtp.assert_not_called()


@pytest.mark.asyncio
async def test_send_notification_invalid_email(email_adapter):
    """Test sending a notification to invalid email."""
    # Setup test data
    user_id = "invalid-email"  # No @ symbol
    message = "Test notification message"
    channel = "email"

    # Mock SMTP server to ensure it's not called
    with patch('smtplib.SMTP') as mock_smtp:
        # Call the method
        result = await email_adapter.send_notification(user_id, message, channel)

        # Verify the result
        assert result is False

        # Verify SMTP was not initialized
        mock_smtp.assert_not_called()


@pytest.mark.asyncio
async def test_send_notification_default_subject(email_adapter):
    """Test sending a notification with default subject."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test notification message"
    channel = "email"

    # Mock SMTP server
    mock_server = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_server

    # Patch SMTP constructor
    with patch('smtplib.SMTP', return_value=mock_context) as _:
        # Call the method with no metadata
        result = await email_adapter.send_notification(user_id, message, channel)

        # Verify the result
        assert result is True

        # Get the message that was sent
        sent_message = mock_server.send_message.call_args[0][0]

        # Verify default subject
        assert sent_message["Subject"] == "Notification from Solana Agent"


@pytest.mark.asyncio
async def test_send_notification_smtp_error(email_adapter):
    """Test sending a notification with SMTP error."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test notification message"
    channel = "email"

    # Patch SMTP constructor to raise exception
    with patch('smtplib.SMTP', side_effect=Exception("SMTP Error")) as _:
        # Call the method
        result = await email_adapter.send_notification(user_id, message, channel)

        # Verify the result
        assert result is False


@pytest.mark.asyncio
async def test_schedule_notification(email_adapter):
    """Test scheduling a notification."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test scheduled message"
    channel = "email"
    schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    metadata = {"subject": "Scheduled Test"}

    # Call the method
    notification_id = await email_adapter.send_scheduled_notification(
        user_id, message, channel, schedule_time, metadata
    )

    # Verify the result
    assert notification_id.startswith("notification_")
    assert notification_id in email_adapter._scheduled_notifications

    # Verify the stored notification details
    notification = email_adapter._scheduled_notifications[notification_id]
    assert notification["user_id"] == user_id
    assert notification["message"] == message
    assert notification["channel"] == channel
    assert notification["schedule_time"] == schedule_time
    assert notification["metadata"] == metadata


@pytest.mark.asyncio
async def test_schedule_notification_default_metadata(email_adapter):
    """Test scheduling a notification without metadata."""
    # Setup test data
    user_id = "user@example.com"
    message = "Test scheduled message"
    channel = "email"
    schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)

    # Call the method without metadata
    notification_id = await email_adapter.send_scheduled_notification(
        user_id, message, channel, schedule_time
    )

    # Verify the result
    assert notification_id in email_adapter._scheduled_notifications

    # Verify empty metadata
    notification = email_adapter._scheduled_notifications[notification_id]
    assert notification["metadata"] == {}


@pytest.mark.asyncio
async def test_cancel_scheduled_notification_existing(email_adapter):
    """Test canceling an existing scheduled notification."""
    # Schedule a notification first
    user_id = "user@example.com"
    message = "Test scheduled message"
    channel = "email"
    schedule_time = datetime.datetime.now() + datetime.timedelta(hours=1)

    notification_id = await email_adapter.send_scheduled_notification(
        user_id, message, channel, schedule_time
    )

    # Now cancel it
    result = await email_adapter.cancel_scheduled_notification(notification_id)

    # Verify the result
    assert result is True
    assert notification_id not in email_adapter._scheduled_notifications


@pytest.mark.asyncio
async def test_cancel_scheduled_notification_nonexistent(email_adapter):
    """Test canceling a non-existent scheduled notification."""
    # Try to cancel a notification that doesn't exist
    result = await email_adapter.cancel_scheduled_notification("nonexistent_id")

    # Verify the result
    assert result is False
