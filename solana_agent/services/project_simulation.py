"""
Project simulation service implementation.

This service simulates project feasibility, timelines, and outcomes
using historical data when available.
"""
import json
from typing import Dict, List, Optional, Any

from solana_agent.interfaces import ProjectSimulationService as ProjectSimulationServiceInterface
from solana_agent.interfaces import TaskPlanningService
from solana_agent.interfaces import LLMProvider
from solana_agent.interfaces import TicketRepository
from solana_agent.domains import TicketStatus


class ProjectSimulationService(ProjectSimulationServiceInterface):
    """Service for simulating project feasibility and requirements using historical data."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        task_planning_service: TaskPlanningService,
        ticket_repository: Optional[TicketRepository] = None,
        nps_repository: Optional[Any] = None,
    ):
        """Initialize the project simulation service.

        Args:
            llm_provider: Provider for language model interactions
            task_planning_service: Service for task planning
            ticket_repository: Optional repository for historical tickets
            nps_repository: Optional repository for NPS survey data
        """
        self.llm_provider = llm_provider
        self.task_planning_service = task_planning_service
        self.ticket_repository = ticket_repository
        self.nps_repository = nps_repository

    async def simulate_project(self, project_description: str) -> Dict[str, Any]:
        """Run a full simulation on a potential project using historical data when available.

        Args:
            project_description: Description of the project to simulate

        Returns:
            Simulation results including complexity, timeline, risks, etc.
        """
        # Get basic complexity assessment
        complexity = await self.task_planning_service._assess_task_complexity(
            project_description
        )

        # Find similar historical projects first
        similar_projects = (
            self._find_similar_projects(project_description, complexity)
            if self.ticket_repository
            else []
        )

        # Get current system load
        system_load = self._analyze_current_load()

        # Perform risk assessment with historical context
        risks = await self._assess_risks(project_description, similar_projects)

        # Estimate timeline with confidence intervals based on history
        timeline = await self._estimate_timeline(
            project_description, complexity, similar_projects
        )

        # Assess resource requirements
        resources = await self._assess_resource_needs(project_description, complexity)

        # Check against current capacity and load
        feasibility = self._assess_feasibility(resources, system_load)

        # Generate recommendation based on all factors
        recommendation = self._generate_recommendation(
            risks, feasibility, similar_projects, system_load
        )

        # Calculate additional insights
        completion_rate = self._calculate_completion_rate(similar_projects)
        avg_satisfaction = self._calculate_avg_satisfaction(similar_projects)

        # Identify top risks for quick reference
        top_risks = []
        if "items" in risks:
            # Sort risks by impact-probability combination
            risk_items = risks.get("items", [])
            impact_map = {"low": 1, "medium": 2, "high": 3}
            for risk in risk_items:
                risk["impact_score"] = impact_map.get(
                    risk.get("impact", "medium").lower(), 2
                )
                risk["probability_score"] = impact_map.get(
                    risk.get("probability", "medium").lower(), 2
                )
                risk["combined_score"] = (
                    risk["impact_score"] * risk["probability_score"]
                )

            # Get top 3 risks by combined score
            sorted_risks = sorted(
                risk_items, key=lambda x: x.get("combined_score", 0), reverse=True
            )
            top_risks = sorted_risks[:3]

        return {
            "complexity": complexity,
            "risks": risks,
            "timeline": timeline,
            "resources": resources,
            "feasibility": feasibility,
            "recommendation": recommendation,
            "top_risks": top_risks,
            "historical_data": {
                "similar_projects_count": len(similar_projects),
                "historical_completion_rate": round(completion_rate * 100, 1),
                "average_satisfaction": round(avg_satisfaction, 1),
                "system_load": round(system_load["load_percentage"], 1),
                "most_similar_project": similar_projects[0]["query"]
                if similar_projects
                else None,
                "satisfaction_trend": "positive"
                if avg_satisfaction > 7
                else "neutral"
                if avg_satisfaction > 5
                else "negative",
            },
        }

    def _find_similar_projects(
        self, project_description: str, complexity: Dict[str, Any]
    ) -> List[Dict]:
        """Find similar historical projects based on semantic similarity and complexity.

        Args:
            project_description: Project description
            complexity: Complexity assessment

        Returns:
            List of similar projects
        """
        if not self.ticket_repository:
            return []

        # Get resolved tickets that were actual projects (higher complexity)
        all_projects = self.ticket_repository.find(
            {
                "status": TicketStatus.RESOLVED,
                "complexity.t_shirt_size": {"$in": ["M", "L", "XL", "XXL"]},
                "complexity.story_points": {"$gte": 5},
            },
            sort_by="resolved_at",
            limit=100,
        )

        if not all_projects:
            return []

        # Compute semantic similarity between current project and historical projects
        try:
            # Create embedding for current project
            embedding = self._get_embedding_for_text(project_description)
            if not embedding:
                return []

            # Get embeddings for historical projects and compute similarity scores
            similar_projects = []
            for ticket in all_projects:
                # Calculate complexity similarity based on t-shirt size and story points
                complexity_similarity = self._calculate_complexity_similarity(
                    complexity, ticket.complexity if ticket.complexity else {}
                )

                # Only include projects with reasonable complexity similarity
                if complexity_similarity > 0.7:
                    similar_projects.append(
                        {
                            "id": ticket.id,
                            "query": ticket.query,
                            "created_at": ticket.created_at,
                            "resolved_at": ticket.resolved_at,
                            "complexity": ticket.complexity,
                            "complexity_similarity": complexity_similarity,
                            "duration_days": (
                                ticket.resolved_at - ticket.created_at
                            ).days,
                        }
                    )

            # Sort by similarity score and return top matches
            return sorted(
                similar_projects, key=lambda x: x["complexity_similarity"], reverse=True
            )[:5]
        except Exception as e:
            print(f"Error finding similar projects: {e}")
            return []

    def _calculate_complexity_similarity(
        self, complexity1: Dict[str, Any], complexity2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two complexity measures.

        Args:
            complexity1: First complexity measure
            complexity2: Second complexity measure

        Returns:
            Similarity score (0-1)
        """
        if not complexity1 or not complexity2:
            return 0.0

        # T-shirt size mapping
        sizes = {"XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}

        # Get t-shirt size values
        size1 = sizes.get(complexity1.get("t_shirt_size", "M"), 3)
        size2 = sizes.get(complexity2.get("t_shirt_size", "M"), 3)

        # Get story point values
        points1 = complexity1.get("story_points", 3)
        points2 = complexity2.get("story_points", 3)

        # Calculate size similarity (normalize by max possible difference)
        size_diff = abs(size1 - size2) / 5.0
        size_similarity = 1 - size_diff

        # Calculate story point similarity (normalize by max common range)
        max_points_diff = 20.0  # Assuming max story points difference we care about
        points_diff = abs(points1 - points2) / max_points_diff
        points_similarity = 1 - min(points_diff, 1.0)

        # Weighted average (give more weight to story points)
        return (size_similarity * 0.4) + (points_similarity * 0.6)

    def _get_embedding_for_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using the LLM provider.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if not available
        """
        try:
            if hasattr(self.llm_provider, "generate_embedding"):
                return self.llm_provider.generate_embedding(text)
            return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def _analyze_current_load(self) -> Dict[str, Any]:
        """Analyze current system load and agent availability.

        Returns:
            Load analysis metrics
        """
        try:
            # Get all AI agents
            ai_agents = self.task_planning_service.agent_service.get_all_ai_agents()
            ai_agent_count = len(ai_agents)

            # Get all human agents
            human_agents = self.task_planning_service.agent_service.get_all_human_agents()
            human_agent_count = len(human_agents)

            # Count available human agents - updated to use proper object access
            available_human_agents = sum(
                1
                for agent in human_agents
                if getattr(agent, "availability_status", None) == "available"
            )

            # Get active tickets using proper ticket statuses
            active_tickets = 0
            if self.ticket_repository:
                active_tickets = self.ticket_repository.count(
                    {
                        "status": {
                            "$in": [
                                TicketStatus.NEW.value,
                                TicketStatus.ASSIGNED.value,
                                TicketStatus.IN_PROGRESS.value,
                                TicketStatus.WAITING_FOR_USER.value,
                                TicketStatus.WAITING_FOR_HUMAN.value,
                            ]
                        }
                    }
                )

            # Calculate load metrics
            total_agents = ai_agent_count + human_agent_count
            if total_agents > 0:
                load_per_agent = active_tickets / total_agents
                load_percentage = min(
                    load_per_agent * 20, 100
                )  # Assuming 5 tickets per agent is 100% load
            else:
                load_percentage = 0

            return {
                "ai_agent_count": ai_agent_count,
                "human_agent_count": human_agent_count,
                "available_human_agents": available_human_agents,
                "active_tickets": active_tickets,
                "load_per_agent": active_tickets / max(total_agents, 1),
                "load_percentage": load_percentage,
            }
        except Exception as e:
            print(f"Error analyzing system load: {e}")
            return {
                "ai_agent_count": 0,
                "human_agent_count": 0,
                "available_human_agents": 0,
                "active_tickets": 0,
                "load_percentage": 0,
            }

    def _calculate_completion_rate(self, similar_projects: List[Dict]) -> float:
        """Calculate a sophisticated completion rate based on historical projects.

        Args:
            similar_projects: List of similar historical projects

        Returns:
            Weighted completion rate (0-1)
        """
        if not similar_projects:
            return 0.0

        # Initialize counters
        successful_projects = 0
        total_weight = 0
        weighted_success = 0

        for project in similar_projects:
            # Calculate similarity weight (more similar projects have higher weight)
            similarity_weight = project.get("complexity_similarity", 0.7)

            # Check if project was completed
            is_completed = "resolved_at" in project and project["resolved_at"]

            # Check timeline adherence if we have the data
            timeline_adherence = 1.0
            if is_completed and "duration_days" in project and "complexity" in project:
                # Get estimated duration from complexity if available
                estimated_days = project.get(
                    "complexity", {}).get("estimated_days", 0)
                if estimated_days > 0:
                    actual_days = project.get("duration_days", 0)
                    # Projects completed on time or early get full score
                    # Projects that took longer get reduced score based on overrun percentage
                    if actual_days <= estimated_days:
                        timeline_adherence = 1.0
                    else:
                        # Max 50% penalty for timeline overruns
                        overrun_factor = min(
                            actual_days / max(estimated_days, 1) - 1, 0.5
                        )
                        timeline_adherence = max(1.0 - overrun_factor, 0.5)

            # Check quality metrics if available
            quality_factor = 1.0
            if self.nps_repository and is_completed and "id" in project:
                surveys = self.nps_repository.db.find(
                    "nps_surveys", {
                        "ticket_id": project["id"], "status": "completed"}
                )
                if surveys:
                    scores = [s.get("score", 0)
                              for s in surveys if "score" in s]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        # Convert 0-10 NPS score to 0.5-1.0 quality factor
                        quality_factor = 0.5 + \
                            (avg_score / 20)  # 10 maps to 1.0

            # Calculate overall project success score
            project_success = is_completed * timeline_adherence * quality_factor

            # Apply weighted scoring
            weighted_success += project_success * similarity_weight
            total_weight += similarity_weight

            # Count basic success for fallback calculation
            if is_completed:
                successful_projects += 1

        # Calculate weighted completion rate if we have weights
        if total_weight > 0:
            return weighted_success / total_weight

        # Fallback to basic completion rate
        return successful_projects / len(similar_projects) if similar_projects else 0

    def _calculate_avg_satisfaction(self, similar_projects: List[Dict]) -> float:
        """Calculate average satisfaction score for similar projects.

        Args:
            similar_projects: List of similar historical projects

        Returns:
            Average satisfaction score (0-10)
        """
        if not similar_projects or not self.nps_repository:
            return 0.0

        scores = []
        for project in similar_projects:
            if project.get("id"):
                # Find NPS survey for this ticket
                surveys = self.nps_repository.db.find(
                    "nps_surveys", {
                        "ticket_id": project["id"], "status": "completed"}
                )
                if surveys:
                    scores.extend([s.get("score", 0)
                                  for s in surveys if "score" in s])

        return sum(scores) / len(scores) if scores else 0

    async def _assess_risks(
        self, project_description: str, similar_projects: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Assess potential risks in the project using historical data.

        Args:
            project_description: Project description
            similar_projects: Optional list of similar historical projects

        Returns:
            Risk assessment results
        """
        # Include historical risk information if available
        historical_context = ""
        if similar_projects:
            historical_context = f"""
            HISTORICAL CONTEXT:
            - {len(similar_projects)} similar projects found in history
            - Average duration: {sum(p.get('duration_days', 0) for p in similar_projects) / len(similar_projects):.1f} days
            - Completion rate: {self._calculate_completion_rate(similar_projects) * 100:.0f}%
            
            Consider this historical data when assessing risks.
            """

        prompt = f"""
        Analyze this potential project and identify risks:
        
        PROJECT: {project_description}
        
        {historical_context}
        
        Please identify:
        1. Technical risks
        2. Timeline risks
        3. Resource/capacity risks
        4. External dependency risks
        5. Risks based on historical performance
        
        For each risk, provide:
        - Description
        - Probability (low/medium/high)
        - Impact (low/medium/high)
        - Potential mitigation strategies
        
        Additionally, provide an overall risk score and classification.
        
        Return as structured JSON.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            "risk_assessor",
            prompt,
            system_prompt="You are an expert risk analyst for software and AI projects with access to historical project data.",
            stream=False,
            model="o3-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
        ):
            response += chunk

        return json.loads(response)

    async def _estimate_timeline(
        self,
        project_description: str,
        complexity: Dict[str, Any],
        similar_projects: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Estimate timeline with confidence intervals using historical data.

        Args:
            project_description: Project description
            complexity: Complexity assessment
            similar_projects: Optional list of similar historical projects

        Returns:
            Timeline estimates
        """
        # Include historical timeline information if available
        historical_context = ""
        if similar_projects:
            # Calculate statistical metrics from similar projects
            durations = [
                p.get("duration_days", 0)
                for p in similar_projects
                if "duration_days" in p
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)

                historical_context = f"""
                HISTORICAL CONTEXT:
                - {len(similar_projects)} similar projects found in history
                - Average duration: {avg_duration:.1f} days
                - Minimum duration: {min_duration} days
                - Maximum duration: {max_duration} days
                
                Use this historical data to inform your timeline estimates.
                """

        prompt = f"""
        Analyze this project and provide timeline estimates:
        
        PROJECT: {project_description}
        
        COMPLEXITY: {json.dumps(complexity)}
        
        {historical_context}
        
        Please provide:
        1. Optimistic timeline (days)
        2. Realistic timeline (days)
        3. Pessimistic timeline (days)
        4. Confidence level in estimate (low/medium/high)
        5. Key factors affecting the timeline
        6. Explanation of how historical data influenced your estimate (if applicable)
        
        Return as structured JSON.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            "timeline_estimator",
            prompt,
            system_prompt="You are an expert project manager skilled at timeline estimation with access to historical project data.",
            stream=False,
            model="o3-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
        ):
            response += chunk

        return json.loads(response)

    async def _assess_resource_needs(
        self, project_description: str, complexity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess resource requirements for the project.

        Args:
            project_description: Project description
            complexity: Complexity assessment

        Returns:
            Resource requirements
        """
        prompt = f"""
        Analyze this project and identify required resources and skills:
        
        PROJECT: {project_description}
        
        COMPLEXITY: {json.dumps(complexity)}
        
        Please identify:
        1. Required agent specializations
        2. Number of agents needed
        3. Required skillsets and expertise levels
        4. External resources or tools needed
        5. Knowledge domains involved
        
        Return as structured JSON.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            "resource_assessor",
            prompt,
            system_prompt="You are an expert resource planner for AI and software projects.",
            stream=False,
            model="o3-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
        ):
            response += chunk

        return json.loads(response)

    def _assess_feasibility(
        self, resource_needs: Dict[str, Any], system_load: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check if we have capacity to take on this project based on current load.

        Args:
            resource_needs: Resource requirements
            system_load: Optional system load metrics

        Returns:
            Feasibility assessment
        """
        # Use empty dict if system_load isn't provided (for test compatibility)
        if system_load is None:
            system_load = {}

        # Get needed specializations
        required_specializations = resource_needs.get(
            "required_specializations", [])

        # Get available agents and their specializations
        available_specializations = set()
        for (
            agent_id,
            specialization,
        ) in self.task_planning_service.agent_service.get_specializations().items():
            available_specializations.add(specialization)

        # Check which required specializations we have
        missing_specializations = []
        for spec in required_specializations:
            found = False
            for avail_spec in available_specializations:
                if (
                    spec.lower() in avail_spec.lower()
                    or avail_spec.lower() in spec.lower()
                ):
                    found = True
                    break
            if not found:
                missing_specializations.append(spec)

        # Calculate expertise coverage
        coverage = 1.0 - (
            len(missing_specializations) /
            max(len(required_specializations), 1)
        )

        # Factor in system load
        load_factor = 1.0
        load_percentage = system_load.get("load_percentage", 0)

        # Adjust load factor based on current system load
        if load_percentage > 90:
            load_factor = 0.3  # Heavily reduce feasibility when system is near capacity
        elif load_percentage > 80:
            load_factor = 0.5  # Significantly reduce feasibility for high load
        elif load_percentage > 60:
            load_factor = 0.8  # Moderately reduce feasibility for medium load

        # Calculate overall feasibility score considering both expertise and load
        feasibility_score = coverage * load_factor * 100

        # Generate feasibility assessment
        return {
            "feasible": feasibility_score > 70,
            "coverage_score": round(coverage * 100, 1),
            "missing_specializations": missing_specializations,
            "available_agents": len(
                self.task_planning_service.agent_service.get_all_ai_agents()
            ),
            "available_specializations": list(available_specializations),
            "system_load_percentage": load_percentage,
            "load_factor": load_factor,
            "assessment": "high"
            if feasibility_score > 80
            else "medium"
            if feasibility_score > 50
            else "low",
            "feasibility_score": round(feasibility_score, 1),
        }

    def _generate_recommendation(
        self,
        risks: Dict[str, Any],
        feasibility: Dict[str, Any],
        similar_projects: Optional[List[Dict]] = None,
        system_load: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an overall recommendation using historical data and current load.

        Args:
            risks: Risk assessment
            feasibility: Feasibility assessment
            similar_projects: Optional list of similar historical projects
            system_load: Optional system load metrics

        Returns:
            Recommendation text
        """
        # Get risk level
        risk_level = risks.get("overall_risk", "medium")

        # Get feasibility assessment - handle both new and old formats for test compatibility
        feasibility_score = feasibility.get(
            "feasibility_score", feasibility.get("coverage_score", 50)
        )
        missing_specializations = feasibility.get(
            "missing_specializations", [])

        # Calculate historical success probability
        historical_context = ""
        if similar_projects:
            historical_success = self._calculate_completion_rate(
                similar_projects)
            historical_context = f" Based on {len(similar_projects)} similar historical projects with a {historical_success*100:.0f}% completion rate,"

        # Factor in system load
        load_context = ""
        if system_load and "load_percentage" in system_load:
            load_percentage = system_load["load_percentage"]
            if load_percentage > 80:
                load_context = f" The system is currently under heavy load ({load_percentage:.0f}%), which may impact delivery."
            elif load_percentage > 60:
                load_context = f" Note that the system is moderately loaded ({load_percentage:.0f}%)."

        # Make recommendation based on feasibility score
        if feasibility_score > 75 and risk_level in ["low", "medium"]:
            return f"RECOMMENDED TO PROCEED:{historical_context} this project has good feasibility ({feasibility_score:.1f}%) with manageable risk level ({risk_level}).{load_context}"

        elif feasibility_score > 60 and risk_level in ["medium", "high"]:
            return f"PROCEED WITH CAUTION:{historical_context} this project has moderate feasibility ({feasibility_score:.1f}%), with {risk_level} risk level that should be mitigated.{load_context}"

        elif feasibility_score <= 60 and len(missing_specializations) > 0:
            return f"NOT RECOMMENDED:{historical_context} this project has low feasibility ({feasibility_score:.1f}%) and requires specializations we lack: {', '.join(missing_specializations)}.{load_context}"

        elif system_load and system_load.get("load_percentage", 0) > 80:
            return f"DELAY RECOMMENDED:{historical_context} while technically possible (feasibility: {feasibility_score:.1f}%), the system is currently under heavy load ({system_load['load_percentage']:.0f}%). Consider scheduling this project for a later time."

        else:
            return f"NEEDS FURTHER ASSESSMENT:{historical_context} with a feasibility score of {feasibility_score:.1f}% and {risk_level} risk level, this project requires more detailed evaluation before proceeding.{load_context}"
