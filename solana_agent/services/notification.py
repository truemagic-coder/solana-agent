
class NotificationService:
    """Service for sending notifications to human agents or users using notification plugins."""

    def __init__(self, human_agent_registry: MongoHumanAgentRegistry, tool_registry=None):
        """Initialize the notification service with a human agent registry."""
        self.human_agent_registry = human_agent_registry
        self.tool_registry = tool_registry

    def send_notification(self, recipient_id: str, message: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Send a notification to a human agent using configured notification channels or legacy handler.
        """
        # Get human agent information
        agent = self.human_agent_registry.get_human_agent(recipient_id)
        if not agent:
            print(f"Cannot send notification: Agent {recipient_id} not found")
            return False

        # BACKWARD COMPATIBILITY: Check for legacy notification handler
        if "notification_handler" in agent and agent["notification_handler"]:
            try:
                metadata = metadata or {}
                agent["notification_handler"](message, metadata)
                return True
            except Exception as e:
                print(
                    f"Error using notification handler for {recipient_id}: {str(e)}")
                return False

        # Get notification channels for this agent
        notification_channels = agent.get("notification_channels", [])
        if not notification_channels:
            print(
                f"No notification channels configured for agent {recipient_id}")
            return False

        # No tool registry available
        if not self.tool_registry:
            print("No tool registry available for notifications")
            return False

        # Try each notification channel until one succeeds
        success = False
        for channel in notification_channels:
            channel_type = channel.get("type")
            channel_config = channel.get("config", {})

            # Execute the notification tool
            try:
                tool_params = {
                    "recipient": recipient_id,
                    "message": message,
                    **channel_config
                }
                if metadata:
                    tool_params["metadata"] = metadata

                tool = self.tool_registry.get_tool(f"notify_{channel_type}")
                if tool:
                    response = tool.execute(**tool_params)
                    if response.get("status") == "success":
                        success = True
                        break
            except Exception as e:
                print(
                    f"Error using notification channel {channel_type} for {recipient_id}: {str(e)}")

        return success

    # Add method needed by tests
    def notify_approvers(self, approver_ids: List[str], message: str, metadata: Dict[str, Any] = None) -> None:
        """
        Send notification to multiple approvers.

        Args:
            approver_ids: List of approver IDs to notify
            message: Notification message content
            metadata: Additional data related to the notification
        """
        for approver_id in approver_ids:
            self.send_notification(approver_id, message, metadata)
