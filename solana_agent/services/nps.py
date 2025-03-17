
class NPSService:
    """Service for managing NPS surveys and ratings."""

    def __init__(
        self, nps_repository: NPSSurveyRepository, ticket_repository: TicketRepository
    ):
        self.nps_repository = nps_repository
        self.ticket_repository = ticket_repository

    def create_survey(self, user_id: str, ticket_id: str, agent_name: str) -> str:
        """Create an NPS survey for a completed ticket."""
        survey = NPSSurvey(
            survey_id=str(uuid.uuid4()),
            user_id=user_id,
            ticket_id=ticket_id,
            agent_name=agent_name,
            status="pending",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

        return self.nps_repository.create(survey)

    def process_response(
        self, survey_id: str, score: int, feedback: Optional[str] = None
    ) -> bool:
        """Process user response to NPS survey."""
        return self.nps_repository.update_response(survey_id, score, feedback)

    def get_agent_score(
        self, agent_name: str, start_date=None, end_date=None
    ) -> Dict[str, Any]:
        """Calculate a comprehensive agent score."""
        # Get NPS metrics
        nps_metrics = self.nps_repository.get_metrics(
            agent_name, start_date, end_date)

        # Get ticket metrics (assuming calculated elsewhere)
        # This is a simplified implementation - in practice we'd get more metrics
        nps_score = nps_metrics.get(
            "avg_score", 0) * 10  # Convert 0-10 to 0-100

        # Calculate overall score - simplified version
        overall_score = nps_score

        return {
            "agent_name": agent_name,
            "overall_score": round(overall_score, 1),
            "rating": self._get_score_rating(overall_score),
            "components": {
                "nps": round(nps_score, 1),
            },
            "metrics": {
                "nps_responses": nps_metrics.get("total_responses", 0),
            },
            "period": {
                "start": start_date.isoformat() if start_date else "All time",
                "end": end_date.isoformat() if end_date else "Present",
            },
        }

    def _get_score_rating(self, score: float) -> str:
        """Convert numerical score to descriptive rating."""
        if score >= 90:
            return "Outstanding"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 50:
            return "Average"
        elif score >= 40:
            return "Below Average"
        elif score >= 30:
            return "Poor"
        else:
            return "Needs Improvement"
