

class ProjectApprovalService:
    """Service for managing human approval of new projects."""

    def __init__(
        self,
        ticket_repository: TicketRepository,
        human_agent_registry: MongoHumanAgentRegistry,
        notification_service: NotificationService = None,
    ):
        self.ticket_repository = ticket_repository
        self.human_agent_registry = human_agent_registry
        self.notification_service = notification_service
        self.approvers = []  # List of human agents with approval privileges

    def register_approver(self, agent_id: str) -> None:
        """Register a human agent as a project approver."""
        if agent_id in self.human_agent_registry.get_all_human_agents():
            self.approvers.append(agent_id)

    async def process_approval(
        self, ticket_id: str, approver_id: str, approved: bool, comments: str = ""
    ) -> None:
        """Process an approval decision."""
        if approver_id not in self.approvers:
            raise ValueError("Not authorized to approve projects")

        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        if approved:
            self.ticket_repository.update(
                ticket_id,
                {
                    "status": TicketStatus.ACTIVE,
                    "approval_status": "approved",
                    "approver_id": approver_id,
                    "approval_comments": comments,
                    "approved_at": datetime.datetime.now(datetime.timezone.utc),
                },
            )
        else:
            self.ticket_repository.update(
                ticket_id,
                {
                    "status": TicketStatus.RESOLVED,
                    "approval_status": "rejected",
                    "approver_id": approver_id,
                    "approval_comments": comments,
                    "rejected_at": datetime.datetime.now(datetime.timezone.utc),
                },
            )

    async def submit_for_approval(self, ticket: Ticket) -> None:
        """Submit a project for human approval."""
        # Update ticket status
        self.ticket_repository.update(
            ticket.id,
            {
                "status": TicketStatus.PENDING,
                "approval_status": "awaiting_approval",
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
            },
        )

        # Notify approvers
        if self.notification_service and self.approvers:
            for approver_id in self.approvers:
                await self.notification_service.send_notification(
                    approver_id,
                    f"New project requires approval: {ticket.query}",
                    {"ticket_id": ticket.id, "type": "approval_request"},
                )
