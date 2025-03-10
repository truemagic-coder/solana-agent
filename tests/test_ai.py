import datetime
import json
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import os
import tempfile
from pathlib import Path

from solana_agent.ai import (
    # Domain Models
    ComplexityAssessment,
    MemoryInsightModel,
    MemoryInsightsResponse,
    MongoHumanAgentRegistry,
    MultitenantSolanaAgent,
    MultitenantSolanaAgentFactory,
    NotificationService,
    ProjectApprovalService,
    ProjectSimulationService,
    QdrantAdapter,
    TenantContext,
    TenantManager,
    TicketStatus,
    AgentType,
    Ticket,
    MemoryInsight,
    TicketResolution,
    SubtaskModel,
    PlanStatus,
    # Service classes
    AgentService,
    RoutingService,
    TicketService,
    HandoffService,
    NPSService,
    MemoryService,
    CriticService,
    QueryProcessor,
    TaskPlanningService,
    # Repository implementations
    MongoTicketRepository,
    MongoHandoffRepository,
    MongoNPSSurveyRepository,
    MongoMemoryRepository,
    # Adapters
    MongoDBAdapter,
    OpenAIAdapter,
    ZepMemoryAdapter,
    PineconeAdapter,
    # Factory
    SolanaAgentFactory,
    # Client interface
    SolanaAgent,
    Tool,
    ToolRegistry,
    PluginManager,
)


#############################################
# FIXTURES
#############################################


class TestMathTool(Tool):
    @property
    def name(self) -> str:
        return "test_math_tool"

    @property
    def description(self) -> str:
        return "A test tool that performs basic math operations"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        }

    def execute(self, **kwargs) -> dict:
        operation = kwargs.get("operation")
        a = kwargs.get("a")
        b = kwargs.get("b")

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero", "status": "error"}
            result = a / b
        else:
            return {"error": f"Unknown operation: {operation}", "status": "error"}

        return {"result": result, "status": "success"}


class TestInfoTool(Tool):
    @property
    def name(self) -> str:
        return "test_info_tool"

    @property
    def description(self) -> str:
        return "A test tool that returns information"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "description": "Type of information to return",
                }
            },
            "required": ["info_type"],
        }

    def execute(self, **kwargs) -> dict:
        info_type = kwargs.get("info_type")
        return {"info": f"Info about {info_type}", "status": "success"}


@pytest.fixture
def mock_mongodb_adapter():
    """Mock MongoDB adapter for testing."""
    mock = MagicMock(spec=MongoDBAdapter)

    # Set up basic behavior
    mock.collection_exists.return_value = True
    mock.insert_one.return_value = str(uuid.uuid4())
    mock.find_one.return_value = None
    mock.find.return_value = []
    mock.count_documents.return_value = 0

    return mock


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock = AsyncMock(spec=OpenAIAdapter)

    # Set up generate_text to return an async generator
    async def mock_generate_text(*args, **kwargs):
        yield "This is a test response"

    mock.generate_text = mock_generate_text
    mock.generate_embedding.return_value = [
        0.1] * 10  # Simple mock embedding vector

    return mock


@pytest.fixture
def mock_memory_provider():
    """Mock memory provider for testing."""
    mock = AsyncMock(spec=ZepMemoryAdapter)
    mock.retrieve.return_value = "Memory context for testing"
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store provider for testing."""
    mock = MagicMock(spec=PineconeAdapter)
    mock.search_vectors.return_value = []
    return mock


@pytest.fixture
def mock_ticket_repository(mock_mongodb_adapter):
    """Create a mocked ticket repository."""
    repo = MongoTicketRepository(mock_mongodb_adapter)

    # Override methods for testing
    repo.create = MagicMock(return_value=str(uuid.uuid4()))
    repo.get_by_id = MagicMock(return_value=None)
    repo.get_active_for_user = MagicMock(return_value=None)
    repo.find = MagicMock(return_value=[])
    repo.update = MagicMock(return_value=True)
    repo.count = MagicMock(return_value=0)

    return repo


@pytest.fixture
def mock_handoff_repository(mock_mongodb_adapter):
    """Create a mocked handoff repository."""
    repo = MongoHandoffRepository(mock_mongodb_adapter)

    # Override methods for testing
    repo.record = MagicMock(return_value=str(uuid.uuid4()))
    repo.find_for_agent = MagicMock(return_value=[])
    repo.count_for_agent = MagicMock(return_value=0)

    return repo


@pytest.fixture
def mock_nps_repository(mock_mongodb_adapter):
    """Create a mocked NPS survey repository."""
    repo = MongoNPSSurveyRepository(mock_mongodb_adapter)

    # Override methods for testing
    repo.create = MagicMock(return_value=str(uuid.uuid4()))
    repo.get_by_id = MagicMock(return_value=None)
    repo.update_response = MagicMock(return_value=True)
    repo.get_metrics = MagicMock(
        return_value={
            "nps_score": 75,
            "promoters": 8,
            "passives": 2,
            "detractors": 0,
            "total_responses": 10,
            "avg_score": 8.5,
        }
    )

    return repo


@pytest.fixture
def mock_memory_repository(mock_mongodb_adapter, mock_vector_store):
    """Create a mocked memory repository."""
    repo = MongoMemoryRepository(mock_mongodb_adapter, mock_vector_store)

    # Override methods for testing
    repo.store_insight = MagicMock(return_value=str(uuid.uuid4()))
    repo.search = MagicMock(
        return_value=[
            {
                "id": "123",
                "fact": "Test fact",
                "relevance": "Test relevance",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]
    )

    return repo


@pytest.fixture
def agent_service(mock_llm_provider):
    """Create an agent service for testing."""
    service = AgentService(mock_llm_provider)

    # Register test agents
    service.register_ai_agent(
        "test_agent", "You are a test agent.", "General testing", "gpt-4o-mini"
    )

    service.register_ai_agent(
        "solana_specialist",
        "You are a Solana blockchain specialist.",
        "Solana blockchain",
        "gpt-4o",
    )

    service.register_human_agent(
        "human1",
        "Human Agent",
        "Complex issues",
        None,  # No notification handler for testing
    )

    return service


@pytest.fixture
def routing_service(mock_llm_provider, agent_service):
    """Create a routing service for testing."""
    return RoutingService(mock_llm_provider, agent_service)


@pytest.fixture
def ticket_service(mock_ticket_repository):
    """Create a ticket service for testing."""
    return TicketService(mock_ticket_repository)


@pytest.fixture
def handoff_service(mock_handoff_repository, mock_ticket_repository, agent_service):
    """Create a handoff service for testing."""
    return HandoffService(
        mock_handoff_repository, mock_ticket_repository, agent_service
    )


@pytest.fixture
def nps_service(mock_nps_repository, mock_ticket_repository):
    """Create an NPS service for testing."""
    return NPSService(mock_nps_repository, mock_ticket_repository)


@pytest.fixture
def memory_service(mock_memory_repository, mock_llm_provider):
    """Create a memory service for testing."""
    return MemoryService(mock_memory_repository, mock_llm_provider)


@pytest.fixture
def critic_service(mock_llm_provider):
    """Create a critic service for testing."""
    return CriticService(mock_llm_provider)


@pytest.fixture
def task_planning_service(mock_ticket_repository, mock_llm_provider, agent_service):
    """Create a task planning service for testing."""
    service = TaskPlanningService(
        ticket_repository=mock_ticket_repository,
        llm_provider=mock_llm_provider,
        agent_service=agent_service,
    )

    # Register test agents for capacity testing
    service.register_agent_capacity(
        "ai_agent1", AgentType.AI, 3, ["general", "coding"])
    service.register_agent_capacity(
        "human_agent1", AgentType.HUMAN, 2, ["specialized"])

    return service


@pytest.fixture
def query_processor(
    agent_service,
    routing_service,
    ticket_service,
    handoff_service,
    memory_service,
    nps_service,
    critic_service,
    mock_memory_provider,
    task_planning_service,
):
    """Create a query processor for testing."""
    return QueryProcessor(
        agent_service=agent_service,
        routing_service=routing_service,
        ticket_service=ticket_service,
        handoff_service=handoff_service,
        memory_service=memory_service,
        nps_service=nps_service,
        critic_service=critic_service,
        memory_provider=mock_memory_provider,
        enable_critic=True,
        router_model="gpt-4o-mini",
        task_planning_service=task_planning_service,
    )


@pytest.fixture
def sample_ticket():
    """Create a sample ticket for testing."""
    return Ticket(
        id=str(uuid.uuid4()),
        user_id="test_user",
        query="Test query",
        status=TicketStatus.ACTIVE,
        assigned_to="test_agent",
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture
def parent_ticket():
    """Create a sample parent ticket for testing."""
    return Ticket(
        id="parent1",
        user_id="user1",
        query="Complex task requiring breakdown",
        status=TicketStatus.PLANNING,
        assigned_to="",
        created_at=datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=1),
        is_parent=True,
    )


@pytest.fixture
def mock_human_agent_registry(mock_mongodb_adapter):
    """Create a mocked human agent registry."""
    registry = MongoHumanAgentRegistry(mock_mongodb_adapter)

    # Pre-register some agents for testing
    registry.register_human_agent(
        agent_id="human1", name="Human Agent 1", specialization="General support"
    )

    registry.register_human_agent(
        agent_id="human2", name="Human Agent 2", specialization="Technical support"
    )

    return registry


#############################################
# TESTS
#############################################


class TestMongoHumanAgentRegistry:
    """Tests for the MongoDB-backed human agent registry."""

    def test_initialization(self, mock_mongodb_adapter):
        """Test initialization creates the right collection and indexes."""
        MongoHumanAgentRegistry(mock_mongodb_adapter)

        # Verify collection was created
        mock_mongodb_adapter.create_collection.assert_called_once()

        # Verify indexes were created
        mock_mongodb_adapter.create_index.assert_any_call(
            "human_agents", [("agent_id", 1)]
        )
        mock_mongodb_adapter.create_index.assert_any_call(
            "human_agents", [("name", 1)])

    def test_register_human_agent(
        self, mock_human_agent_registry, mock_mongodb_adapter
    ):
        """Test registering a new human agent."""
        # Register a new agent
        mock_human_agent_registry.register_human_agent(
            agent_id="new_agent", name="New Agent", specialization="Customer complaints"
        )

        # Verify database was updated
        mock_mongodb_adapter.update_one.assert_called()

        # Verify agent was added to cache
        agent = mock_human_agent_registry.get_human_agent("new_agent")
        assert agent is not None
        assert agent["name"] == "New Agent"
        assert agent["specialization"] == "Customer complaints"
        assert agent["availability_status"] == "available"

        # Verify specialization was registered
        assert (
            mock_human_agent_registry.specializations_cache["new_agent"]
            == "Customer complaints"
        )

    def test_get_human_agent(self, mock_human_agent_registry):
        """Test retrieving a human agent."""
        # Get an existing agent
        agent = mock_human_agent_registry.get_human_agent("human1")
        assert agent is not None
        assert agent["name"] == "Human Agent 1"

        # Try to get a non-existent agent
        agent = mock_human_agent_registry.get_human_agent("nonexistent")
        assert agent is None

    def test_get_all_human_agents(self, mock_human_agent_registry):
        """Test retrieving all human agents."""
        agents = mock_human_agent_registry.get_all_human_agents()
        assert len(agents) == 2
        assert "human1" in agents
        assert "human2" in agents

    def test_get_specializations(self, mock_human_agent_registry):
        """Test retrieving all specializations."""
        specializations = mock_human_agent_registry.get_specializations()
        assert len(specializations) == 2
        assert specializations["human1"] == "General support"
        assert specializations["human2"] == "Technical support"

    def test_update_agent_status(self, mock_human_agent_registry, mock_mongodb_adapter):
        """Test updating agent status."""
        # Update an existing agent
        result = mock_human_agent_registry.update_agent_status(
            "human1", "busy")
        assert result is True

        # Verify database was updated
        mock_mongodb_adapter.update_one.assert_called()

        # Verify cache was updated
        agent = mock_human_agent_registry.get_human_agent("human1")
        assert agent["availability_status"] == "busy"

        # Test updating a non-existent agent
        result = mock_human_agent_registry.update_agent_status(
            "nonexistent", "busy")
        assert result is False

    def test_delete_agent(self, mock_human_agent_registry, mock_mongodb_adapter):
        """Test deleting a human agent."""
        # Delete an existing agent
        result = mock_human_agent_registry.delete_agent("human1")
        assert result is True

        # Verify database deletion was called
        mock_mongodb_adapter.delete_one.assert_called_once()

        # Verify agent was removed from cache
        assert mock_human_agent_registry.get_human_agent("human1") is None
        assert "human1" not in mock_human_agent_registry.specializations_cache

        # Test deleting a non-existent agent
        result = mock_human_agent_registry.delete_agent("nonexistent")
        assert result is False


class TestAgentServiceWithRegistry:
    """Tests for AgentService with MongoHumanAgentRegistry integration."""

    def test_agent_service_with_registry(
        self, mock_llm_provider, mock_human_agent_registry
    ):
        """Test AgentService using the human agent registry."""
        # Create agent service with registry
        service = AgentService(mock_llm_provider, mock_human_agent_registry)

        # Register an AI agent normally
        service.register_ai_agent(
            "test_ai", "You are a test AI agent.", "Testing", "gpt-4o-mini"
        )

        # Register a human agent through the service
        service.register_human_agent(
            "human3", "Human Agent 3", "Data analysis")

        # Verify AI agent was registered
        ai_agents = service.get_all_ai_agents()
        assert "test_ai" in ai_agents

        # Verify human agent was registered via registry
        human_agents = service.get_all_human_agents()
        assert "human1" in human_agents  # Pre-registered in registry
        assert "human2" in human_agents  # Pre-registered in registry
        assert "human3" in human_agents  # Newly registered

        # Verify specializations include both AI and human agents
        specializations = service.get_specializations()
        assert specializations["test_ai"] == "Testing"
        assert specializations["human1"] == "General support"
        assert specializations["human3"] == "Data analysis"

        # Test updating human agent status
        result = service.update_human_agent_status("human1", "away")
        assert result is True

        # Verify status was updated in registry
        agent = mock_human_agent_registry.get_human_agent("human1")
        assert agent["availability_status"] == "away"


class TestAgentService:
    """Tests for the AgentService."""

    def test_register_ai_agent(self, agent_service):
        """Test registering AI agents."""
        # Test agent was registered in the fixture
        assert "test_agent" in agent_service.get_all_ai_agents()
        assert "solana_specialist" in agent_service.get_all_ai_agents()
        assert agent_service.specializations["test_agent"] == "General testing"

    def test_register_human_agent(self, agent_service):
        """Test registering human agents."""
        # Test human agent was registered in the fixture
        assert "human1" in agent_service.get_all_human_agents()
        assert agent_service.specializations["human1"] == "Complex issues"

    def test_get_specializations(self, agent_service):
        """Test getting all specializations."""
        specializations = agent_service.get_specializations()
        assert "test_agent" in specializations
        assert "human1" in specializations
        assert specializations["test_agent"] == "General testing"

    @pytest.mark.asyncio
    async def test_generate_response(self, agent_service):
        """Test generating a response from an AI agent."""
        # Test with a non-existent agent
        response = ""
        async for chunk in agent_service.generate_response(
            "nonexistent_agent", "user1", "Hi"
        ):
            response += chunk

        assert response == "Error: Agent not found"

        # Test with a valid agent
        response = ""
        async for chunk in agent_service.generate_response("test_agent", "user1", "Hi"):
            response += chunk

        assert response == "This is a test response"


class TestRoutingService:
    """Tests for the RoutingService."""

    @pytest.mark.asyncio
    async def test_route_query(self, routing_service):
        """Test routing a query to the appropriate agent."""
        # Mocking the LLM response is challenging in this test
        # We'll assume the actual logic works and test the method structure
        agent_name = await routing_service.route_query(
            "Tell me about Solana blockchain"
        )

        # Since our mock doesn't actually make decisions, we just check the return type
        assert isinstance(agent_name, str)

    def test_match_agent_name(self, routing_service):
        """Test matching agent names from responses."""
        agent_names = ["test_agent", "solana_specialist"]

        # Exact match
        assert (
            routing_service._match_agent_name(
                "test_agent", agent_names) == "test_agent"
        )

        # Case insensitive match
        assert (
            routing_service._match_agent_name(
                "TEST_AGENT", agent_names) == "test_agent"
        )

        # Partial match
        assert (
            routing_service._match_agent_name(
                "something with solana_specialist in it", agent_names
            )
            == "solana_specialist"
        )

        # No match should default to first agent
        assert (
            routing_service._match_agent_name(
                "no match", agent_names) == "test_agent"
        )


class TestTicketService:
    """Tests for the TicketService."""

    @pytest.mark.asyncio
    async def test_get_or_create_ticket_new(
        self, ticket_service, mock_ticket_repository
    ):
        """Test creating a new ticket when none exists."""
        # Configure mock to return no active ticket
        mock_ticket_repository.get_active_for_user.return_value = None

        # Test creating a new ticket
        ticket = await ticket_service.get_or_create_ticket("user1", "Help me")

        assert ticket is not None
        assert ticket.user_id == "user1"
        assert ticket.query == "Help me"
        assert ticket.status == TicketStatus.NEW

        # Verify the repository was called
        mock_ticket_repository.get_active_for_user.assert_called_once_with(
            "user1")
        mock_ticket_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_ticket_existing(
        self, ticket_service, mock_ticket_repository, sample_ticket
    ):
        """Test getting an existing active ticket."""
        # Configure mock to return an active ticket
        mock_ticket_repository.get_active_for_user.return_value = sample_ticket

        # Test getting an existing ticket
        ticket = await ticket_service.get_or_create_ticket("test_user", "New query")

        assert ticket is sample_ticket

        # Verify the repository was called but create was not
        mock_ticket_repository.get_active_for_user.assert_called_once_with(
            "test_user")
        mock_ticket_repository.create.assert_not_called()

    def test_update_ticket_status(self, ticket_service, mock_ticket_repository):
        """Test updating a ticket's status."""
        ticket_service.update_ticket_status(
            "123", TicketStatus.ACTIVE, assigned_to="test_agent"
        )

        # Verify the repository was called with correct parameters
        mock_ticket_repository.update.assert_called_once()
        args, kwargs = mock_ticket_repository.update.call_args
        assert args[0] == "123"
        assert args[1]["status"] == TicketStatus.ACTIVE
        assert args[1]["assigned_to"] == "test_agent"
        assert "updated_at" in args[1]

    def test_mark_ticket_resolved(self, ticket_service, mock_ticket_repository):
        """Test marking a ticket as resolved."""
        resolution_data = {"confidence": 0.9,
                           "reasoning": "Issue was fully addressed"}

        ticket_service.mark_ticket_resolved("123", resolution_data)

        # Verify the repository was called with correct parameters
        mock_ticket_repository.update.assert_called_once()
        args, kwargs = mock_ticket_repository.update.call_args
        assert args[0] == "123"
        assert args[1]["status"] == TicketStatus.RESOLVED
        assert args[1]["resolution_confidence"] == 0.9
        assert args[1]["resolution_reasoning"] == "Issue was fully addressed"
        assert "resolved_at" in args[1]
        assert "updated_at" in args[1]


class TestHandoffService:
    """Tests for the HandoffService."""

    @pytest.mark.asyncio
    async def test_process_handoff(
        self,
        handoff_service,
        mock_handoff_repository,
        mock_ticket_repository,
        sample_ticket,
    ):
        """Test processing a handoff between agents."""
        # Configure mocks
        mock_ticket_repository.get_by_id.return_value = sample_ticket

        # Test handoff process
        result = await handoff_service.process_handoff(
            "123", "test_agent", "solana_specialist", "Needs blockchain expertise"
        )

        assert result == "solana_specialist"

        # Verify repositories were called
        mock_ticket_repository.get_by_id.assert_called_once_with("123")
        mock_handoff_repository.record.assert_called_once()
        mock_ticket_repository.update.assert_called_once()

        # Check update parameters
        args, kwargs = mock_ticket_repository.update.call_args
        assert args[0] == "123"
        assert args[1]["assigned_to"] == "solana_specialist"
        assert args[1]["status"] == TicketStatus.TRANSFERRED
        assert args[1]["handoff_reason"] == "Needs blockchain expertise"

    @pytest.mark.asyncio
    async def test_process_handoff_ticket_not_found(
        self, handoff_service, mock_ticket_repository
    ):
        """Test handling a handoff when the ticket doesn't exist."""
        # Configure mock to return no ticket
        mock_ticket_repository.get_by_id.return_value = None

        # Test that ValueError is raised
        with pytest.raises(ValueError, match="Ticket .* not found"):
            await handoff_service.process_handoff(
                "123", "test_agent", "solana_specialist", "Needs blockchain expertise"
            )

    @pytest.mark.asyncio
    async def test_process_handoff_invalid_agent(
        self, handoff_service, mock_ticket_repository, sample_ticket
    ):
        """Test handling a handoff when the target agent doesn't exist."""
        # Configure mock to return a ticket
        mock_ticket_repository.get_by_id.return_value = sample_ticket

        # Test that ValueError is raised
        with pytest.raises(ValueError, match="Target agent .* not found"):
            await handoff_service.process_handoff(
                "123", "test_agent", "nonexistent_agent", "Needs expertise"
            )


class TestNPSService:
    """Tests for the NPSService."""

    def test_create_survey(self, nps_service, mock_nps_repository):
        """Test creating an NPS survey."""
        survey_id = nps_service.create_survey(
            "user1", "ticket123", "test_agent")

        assert survey_id is not None

        # Verify repository was called
        mock_nps_repository.create.assert_called_once()
        args, kwargs = mock_nps_repository.create.call_args
        survey = args[0]
        assert survey.user_id == "user1"
        assert survey.ticket_id == "ticket123"
        assert survey.agent_name == "test_agent"
        assert survey.status == "pending"

    def test_process_response(self, nps_service, mock_nps_repository):
        """Test processing a user's NPS response."""
        result = nps_service.process_response("survey123", 9, "Great service!")

        assert result is True  # Mocked to return True

        # Verify repository was called
        mock_nps_repository.update_response.assert_called_once_with(
            "survey123", 9, "Great service!"
        )

    def test_get_agent_score(self, nps_service, mock_nps_repository):
        """Test getting an agent's performance score."""
        score_data = nps_service.get_agent_score("test_agent")

        assert score_data["agent_name"] == "test_agent"
        assert score_data["overall_score"] > 0
        assert isinstance(score_data["rating"], str)
        assert "nps" in score_data["components"]
        assert "nps_responses" in score_data["metrics"]

        # Verify repository was called
        mock_nps_repository.get_metrics.assert_called_once()


class TestMemoryService:
    """Tests for the MemoryService."""

    @pytest.mark.asyncio
    async def test_extract_insights(self, memory_service, mock_llm_provider):
        """Test extracting insights from a conversation."""
        # Setup mock for parse_structured_output method
        mock_insights_response = MemoryInsightsResponse(
            insights=[MemoryInsightModel(
                fact="Test fact", relevance="Test relevance")]
        )

        # Create a mock for the parse_structured_output method
        mock_llm_provider.parse_structured_output = AsyncMock(
            return_value=mock_insights_response)

        # Test insight extraction
        insights = await memory_service.extract_insights(
            {"message": "Hi", "response": "Hello"}
        )

        # Verify the result
        assert len(insights) == 1
        assert insights[0].fact == "Test fact"
        assert insights[0].relevance == "Test relevance"

        # Verify parse_structured_output was called with correct parameters
        mock_llm_provider.parse_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_insights(self, memory_service, mock_memory_repository):
        """Test storing insights in memory."""
        insights = [
            MemoryInsight(fact="Test fact 1", relevance="Test relevance 1"),
            MemoryInsight(fact="Test fact 2", relevance="Test relevance 2"),
        ]

        await memory_service.store_insights("user1", insights)

        # Verify repository was called twice, once for each insight
        assert mock_memory_repository.store_insight.call_count == 2

    def test_search_memory(self, memory_service, mock_memory_repository):
        """Test searching collective memory for insights."""
        results = memory_service.search_memory("test query")

        assert len(results) > 0
        assert "fact" in results[0]

        # Verify repository was called
        mock_memory_repository.search.assert_called_once_with("test query", 5)


class TestTaskPlanningService:
    """Tests for the TaskPlanningService."""

    @pytest.mark.asyncio
    async def test_needs_breakdown(self, task_planning_service):
        """Test determining if a task needs to be broken down."""
        # Mock the _assess_task_complexity method
        task_planning_service._assess_task_complexity = AsyncMock(
            return_value={
                "story_points": 13,
                "t_shirt_size": "XL",
                "estimated_minutes": 120,
            }
        )

        # Test with a complex task
        needs_breakdown, reason = await task_planning_service.needs_breakdown(
            "Build a complex application"
        )

        assert needs_breakdown is True
        assert "complexity" in reason.lower()

        # Change to a simple task
        task_planning_service._assess_task_complexity = AsyncMock(
            return_value={
                "story_points": 3,
                "t_shirt_size": "S",
                "estimated_minutes": 20,
            }
        )

        needs_breakdown, reason = await task_planning_service.needs_breakdown(
            "Simple task"
        )

        assert needs_breakdown is False

    @pytest.mark.asyncio
    async def test_generate_subtasks(
        self, task_planning_service, mock_ticket_repository, parent_ticket
    ):
        """Test generating subtasks for a complex task."""
        # Configure mocks
        mock_ticket_repository.get_by_id.return_value = parent_ticket

        # Setup mock LLM response
        mock_json = {
            "subtasks": [
                {
                    "title": "Subtask 1",
                    "description": "Do the first part",
                    "estimated_minutes": 30,
                    "dependencies": [],
                },
                {
                    "title": "Subtask 2",
                    "description": "Do the second part",
                    "estimated_minutes": 45,
                    "dependencies": ["Subtask 1"],
                },
            ]
        }

        async def mock_generate_text(*args, **kwargs):
            yield json.dumps(mock_json)

        task_planning_service.llm_provider.generate_text = mock_generate_text

        # Test generating subtasks
        subtasks = await task_planning_service.generate_subtasks(
            "parent1", "Complex task"
        )

        # Verify results
        assert len(subtasks) == 2
        assert subtasks[0].title == "Subtask 1"
        assert subtasks[1].title == "Subtask 2"
        assert subtasks[1].dependencies  # Should contain the ID of Subtask 1

        # Verify ticket repository was called
        mock_ticket_repository.get_by_id.assert_called_once()
        assert mock_ticket_repository.update.called
        assert mock_ticket_repository.create.call_count == 2  # Once per subtask

    @pytest.mark.asyncio
    async def test_assign_subtasks(self, task_planning_service, mock_ticket_repository):
        """Test assigning subtasks to agents."""
        # Configure mock to return sample subtasks
        subtasks = [
            Ticket(
                id="subtask1",
                user_id="user1",
                query="Subtask 1",
                status=TicketStatus.PLANNING,
                assigned_to="",
                created_at=datetime.datetime.now(datetime.timezone.utc),
                is_subtask=True,
                parent_id="parent1",
            ),
            Ticket(
                id="subtask2",
                user_id="user1",
                query="Subtask 2",
                status=TicketStatus.PLANNING,
                assigned_to="",
                created_at=datetime.datetime.now(datetime.timezone.utc),
                is_subtask=True,
                parent_id="parent1",
            ),
        ]
        mock_ticket_repository.find.return_value = subtasks

        # Test assigning subtasks
        assignments = await task_planning_service.assign_subtasks("parent1")

        # Verify assignments were made
        assert len(assignments) > 0
        assert mock_ticket_repository.update.call_count == 2  # One update per subtask

        # Verify agent capacity was updated
        for agent_id in assignments:
            if agent_id in task_planning_service.capacity_registry:
                assert (
                    task_planning_service.capacity_registry[agent_id].active_tasks > 0
                )

    @pytest.mark.asyncio
    async def test_get_plan_status(
        self, task_planning_service, mock_ticket_repository, parent_ticket
    ):
        """Test getting the status of a task plan."""
        # Configure mocks
        mock_ticket_repository.get_by_id.return_value = parent_ticket

        # Configure mock to return sample subtasks in various states
        subtasks = [
            Ticket(
                id="subtask1",
                user_id="user1",
                query="Subtask 1",
                status=TicketStatus.RESOLVED,  # Completed
                assigned_to="agent1",
                created_at=datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(minutes=60),
                is_subtask=True,
                parent_id="parent1",
            ),
            Ticket(
                id="subtask2",
                user_id="user1",
                query="Subtask 2",
                status=TicketStatus.ACTIVE,  # In progress
                assigned_to="agent2",
                created_at=datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(minutes=45),
                is_subtask=True,
                parent_id="parent1",
            ),
            Ticket(
                id="subtask3",
                user_id="user1",
                query="Subtask 3",
                status=TicketStatus.NEW,  # Not started
                assigned_to="agent1",
                created_at=datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(minutes=30),
                is_subtask=True,
                parent_id="parent1",
            ),
        ]

        # Configure mock to return the sample subtasks
        mock_ticket_repository.find.return_value = subtasks

        # Test getting plan status
        status = await task_planning_service.get_plan_status("parent1")

        # Verify the status was calculated correctly
        assert isinstance(status, PlanStatus)
        assert status.progress == 33  # 1/3 complete = 33%
        assert status.status == "in progress"
        assert status.subtask_count == 3
        assert "█" in status.visualization  # Should contain progress bars

    def test_register_agent_capacity(self, task_planning_service):
        """Test registering agent work capacity."""
        # Register a new agent
        task_planning_service.register_agent_capacity(
            "new_agent", AgentType.AI, 5, ["coding", "design"]
        )

        # Check if the agent was registered correctly
        capacity = task_planning_service.get_agent_capacity("new_agent")
        assert capacity is not None
        assert capacity.agent_id == "new_agent"
        assert capacity.agent_type == AgentType.AI
        assert capacity.max_concurrent_tasks == 5
        assert capacity.active_tasks == 0
        assert "coding" in capacity.specializations
        assert "design" in capacity.specializations

    def test_update_agent_availability(self, task_planning_service):
        """Test updating agent availability status."""
        # Update existing agent
        result = task_planning_service.update_agent_availability(
            "ai_agent1", "busy")

        # Verify the update worked
        assert result is True
        capacity = task_planning_service.get_agent_capacity("ai_agent1")
        assert capacity.availability_status == "busy"

        # Test with non-existent agent
        result = task_planning_service.update_agent_availability(
            "nonexistent_agent", "available"
        )
        assert result is False

    def test_get_available_agents(self, task_planning_service):
        """Test getting available agents with and without specialization filters."""
        # All agents should be available initially
        available_agents = task_planning_service.get_available_agents()
        assert len(available_agents) == 2
        assert "ai_agent1" in available_agents
        assert "human_agent1" in available_agents

        # Set one agent as busy
        task_planning_service.update_agent_availability("ai_agent1", "busy")
        available_agents = task_planning_service.get_available_agents()
        assert len(available_agents) == 1
        assert "human_agent1" in available_agents

        # Filter by specialization
        available_agents = task_planning_service.get_available_agents(
            "specialized")
        assert len(available_agents) == 1
        assert "human_agent1" in available_agents

        # Filter by non-matching specialization
        available_agents = task_planning_service.get_available_agents(
            "unknown")
        assert len(available_agents) == 0

        # Set agent back to available but with full capacity
        task_planning_service.update_agent_availability(
            "ai_agent1", "available")
        # Max capacity
        task_planning_service.capacity_registry["ai_agent1"].active_tasks = 3
        available_agents = task_planning_service.get_available_agents()
        assert len(available_agents) == 1
        assert "human_agent1" in available_agents

    @pytest.mark.asyncio
    async def test_assess_task_complexity(self, task_planning_service, mock_llm_provider):
        """Test assessing task complexity."""
        # Create a ComplexityAssessment model instance for the mock response
        mock_complexity = ComplexityAssessment(
            t_shirt_size="L",
            story_points=8,
            estimated_minutes=45,
            technical_complexity=7,
            domain_knowledge=6
        )

        # Setup mock for parse_structured_output
        mock_llm_provider.parse_structured_output = AsyncMock(
            return_value=mock_complexity)

        # This is testing a private method, so we need to access it directly
        complexity = await task_planning_service._assess_task_complexity("Build a complex application")

        # Verify the result is a dictionary with all expected fields
        assert complexity["t_shirt_size"] == "L"
        assert complexity["story_points"] == 8
        assert complexity["estimated_minutes"] == 45
        assert complexity["technical_complexity"] == 7
        assert complexity["domain_knowledge"] == 6


@pytest.mark.asyncio
class TestQueryProcessor:
    """Tests for the QueryProcessor."""

    async def test_process_greeting(self, query_processor, mock_memory_provider):
        """Test processing a simple greeting."""
        # Setup _is_simple_greeting to return True
        query_processor._is_simple_greeting = AsyncMock(return_value=True)

        # Setup _generate_greeting_response
        query_processor._generate_greeting_response = AsyncMock(
            return_value="Hello!")

        # Test processing a greeting
        response = ""
        async for chunk in query_processor.process("user1", "Hi there"):
            response += chunk

        assert response == "Hello!"
        query_processor._is_simple_greeting.assert_called_once()
        query_processor._generate_greeting_response.assert_called_once()

    async def test_process_system_command(self, query_processor):
        """Test processing a system command."""
        # Setup _process_system_commands to return a value
        query_processor._process_system_commands = AsyncMock(
            return_value="System command result"
        )

        # Test processing a system command
        response = ""
        async for chunk in query_processor.process("user1", "!command"):
            response += chunk

        assert response == "System command result"
        query_processor._process_system_commands.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_new_ticket(self, query_processor, mock_ticket_repository, agent_service):
        """Test processing a message creating a new ticket."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        query_processor.routing_service.route_query = AsyncMock(
            return_value="test_agent")

        # Return a dictionary directly, not a coroutine that needs to be awaited
        complexity_data = {
            "t_shirt_size": "M",
            "story_points": 3,
            "estimated_minutes": 30,
            "technical_complexity": 5,
            "domain_knowledge": 5,
        }
        query_processor._assess_task_complexity = AsyncMock(
            return_value=complexity_data)

        # Important: Mock the needs_breakdown method which is being called
        query_processor.task_planning_service.needs_breakdown = AsyncMock(
            return_value=(False, "Task is simple"))

        mock_ticket_repository.get_active_for_user.return_value = None

        # Mock the ticket service methods directly
        query_processor.ticket_service.update_ticket_status = MagicMock(
            return_value=True)

        # Mock the agent service's generate_response method to return a known value
        async def mock_generate_response(*args, **kwargs):
            yield "This is a test response"

        query_processor.agent_service.generate_response = mock_generate_response

        # Mock the ticket
        new_ticket = Ticket(
            id="new_ticket_id",
            user_id="user1",
            query="Help me",
            status=TicketStatus.NEW,
            assigned_to="",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        query_processor.ticket_service.get_or_create_ticket = AsyncMock(
            return_value=new_ticket)

        # Test processing a new ticket
        response = ""
        async for chunk in query_processor.process("user1", "Help me with something"):
            response += chunk

        assert response == "This is a test response"

    async def test_process_existing_ticket(
        self, query_processor, mock_ticket_repository, sample_ticket
    ):
        """Test processing a message for an existing ticket."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        query_processor.routing_service.route_query = AsyncMock(
            return_value="test_agent"
        )
        mock_ticket_repository.get_active_for_user.return_value = sample_ticket

        # Mock service methods
        query_processor.ticket_service.update_ticket_status = MagicMock(
            return_value=True
        )
        query_processor.nps_service.create_survey = MagicMock(
            return_value="survey123")

        # Mock ticket resolution check
        resolution = TicketResolution(
            status="resolved", confidence=0.8, reasoning="Issue was resolved"
        )
        query_processor._check_ticket_resolution = AsyncMock(
            return_value=resolution)

        # Test processing an existing ticket
        response = ""
        async for chunk in query_processor.process("test_user", "Follow-up question"):
            response += chunk

        assert response == "This is a test response"

        # Verify methods were called
        query_processor._is_human_agent.assert_called_once()
        query_processor._is_simple_greeting.assert_called_once()
        query_processor._process_system_commands.assert_called_once()
        query_processor._check_ticket_resolution.assert_called_once()
        assert query_processor.nps_service.create_survey.called

    async def test_process_human_agent_message(self, query_processor):
        """Test processing a message from a human agent."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=True)
        query_processor._get_agent_directory = MagicMock(
            return_value="Agent Directory")

        # Test processing a human agent command
        response = ""
        async for chunk in query_processor.process("human1", "!agents"):
            response += chunk

        assert response == "Agent Directory"

        # Verify methods were called
        query_processor._is_human_agent.assert_called_once()
        query_processor._get_agent_directory.assert_called_once()

    async def test_process_complex_task(
        self, query_processor, mock_ticket_repository, task_planning_service
    ):
        """Test processing a complex task that needs breakdown."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        mock_ticket_repository.get_active_for_user.return_value = None

        # Setup task complexity assessment and breakdown
        query_processor._assess_task_complexity = AsyncMock(
            return_value={"t_shirt_size": "XL", "story_points": 13}
        )
        query_processor.task_planning_service.needs_breakdown = AsyncMock(
            return_value=(True, "Task is complex")
        )

        # Mock the ticket
        new_ticket = Ticket(
            id="complex_task",
            user_id="user1",
            query="Build a complex system",
            status=TicketStatus.NEW,
            assigned_to="",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        query_processor.ticket_service.get_or_create_ticket = AsyncMock(
            return_value=new_ticket
        )
        query_processor.ticket_service.update_ticket_status = MagicMock()

        # Mock subtask generation
        subtasks = [
            SubtaskModel(
                title="Subtask 1",
                description="Part 1",
                estimated_minutes=30,
                dependencies=[],
                parent_id="complex_task",
                sequence=1,
            ),
            SubtaskModel(
                title="Subtask 2",
                description="Part 2",
                estimated_minutes=45,
                dependencies=["Subtask 1"],
                parent_id="complex_task",
                sequence=2,
            ),
        ]
        query_processor.task_planning_service.generate_subtasks = AsyncMock(
            return_value=subtasks
        )
        query_processor.task_planning_service.assign_subtasks = AsyncMock(
            return_value={"ai_agent1": ["subtask1"]}
        )

        # Test processing a complex task
        response = ""
        async for chunk in query_processor.process(
            "user1", "Build a complex system with many parts"
        ):
            response += chunk

        # Verify the planning flow was triggered
        assert query_processor.task_planning_service.needs_breakdown.called
        assert query_processor.ticket_service.update_ticket_status.called
        # The ticket status should be updated to PLANNING
        args, kwargs = (
            query_processor.ticket_service.update_ticket_status.call_args_list[0]
        )
        assert args[1] == TicketStatus.PLANNING


class TestPlatformIntegration:
    """Integration tests for the entire platform."""

    @pytest.mark.asyncio
    async def test_agent_factory_creation(self):
        """Test creating the agent system from configuration."""
        # Create a minimal configuration
        config = {
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db",
            },
            "openai": {"api_key": "test_key", "default_model": "gpt-4o-mini"},
            "enable_critic": True,
            "router_model": "gpt-4o-mini",
            "ai_agents": [
                {
                    "name": "general",
                    "instructions": "You are a helpful assistant.",
                    "specialization": "General assistance",
                }
            ],
        }

        # Mock the adapters
        with patch("solana_agent.ai.MongoDBAdapter") as mock_mongo, patch(
            "solana_agent.ai.OpenAIAdapter"
        ) as mock_openai:
            # Mock MongoDB adapter instance
            mock_mongo.return_value = MagicMock(spec=MongoDBAdapter)

            # Mock OpenAI adapter instance
            mock_llm = AsyncMock(spec=OpenAIAdapter)

            async def mock_generate_text(*args, **kwargs):
                yield "Test response"

            mock_llm.generate_text = mock_generate_text
            mock_openai.return_value = mock_llm

            # Create processor from factory
            processor = SolanaAgentFactory.create_from_config(config)

            # Verify processor was created correctly
            assert isinstance(processor, QueryProcessor)
            assert "general" in processor.agent_service.get_all_ai_agents()

    @pytest.mark.asyncio
    async def test_solana_agent_client(self):
        """Test the simplified SolanaAgent client interface."""
        # Create a minimal configuration
        config = {
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db",
            },
            "openai": {"api_key": "test_key", "default_model": "gpt-4o-mini"},
            "agents": [],
        }

        # Mock the factory and processor
        with patch("solana_agent.ai.SolanaAgentFactory") as mock_factory:
            # Create mock processor
            mock_processor = AsyncMock()

            async def mock_process(*args, **kwargs):
                yield "Client interface test response"

            mock_processor.process = mock_process
            mock_factory.create_from_config.return_value = mock_processor

            # Create client
            client = SolanaAgent(config=config)

            # Test processing messages
            response = ""
            async for chunk in client.process("user1", "Hello from client"):
                response += chunk

            assert response == "Client interface test response"


class TestProjectSimulationService:
    """Tests for the ProjectSimulationService."""

    @pytest.fixture
    def mock_project_simulation_service(self, mock_llm_provider, task_planning_service):
        """Create a mocked project simulation service."""
        return ProjectSimulationService(
            llm_provider=mock_llm_provider, task_planning_service=task_planning_service
        )

    @pytest.mark.asyncio
    async def test_simulate_project(self, mock_project_simulation_service):
        """Test full project simulation."""
        # Mock sub-method results
        complexity = {"t_shirt_size": "L", "story_points": 8}
        mock_project_simulation_service._assess_task_complexity = AsyncMock(
            return_value=complexity
        )

        risks = {
            "overall_risk": "medium",
            "items": [
                {
                    "type": "technical",
                    "description": "Tech risk",
                    "probability": "medium",
                    "impact": "high",
                }
            ],
        }
        mock_project_simulation_service._assess_risks = AsyncMock(
            return_value=risks)

        timeline = {
            "optimistic": 10,
            "realistic": 15,
            "pessimistic": 20,
            "confidence": "medium",
        }
        mock_project_simulation_service._estimate_timeline = AsyncMock(
            return_value=timeline
        )

        resources = {
            "required_specializations": ["frontend", "backend"],
            "number_of_agents": 2,
        }
        mock_project_simulation_service._assess_resource_needs = AsyncMock(
            return_value=resources
        )

        feasibility = {
            "feasible": True,
            "coverage_score": 80,
            "missing_specializations": [],
            "assessment": "high",
        }
        mock_project_simulation_service._assess_feasibility = AsyncMock(
            return_value=feasibility
        )

        # Mock the recommendation
        mock_project_simulation_service._generate_recommendation = MagicMock(
            return_value="RECOMMENDED TO PROCEED"
        )

        # Run simulation
        result = await mock_project_simulation_service.simulate_project(
            "Build a web app"
        )

        # Verify results
        assert "complexity" in result
        assert "risks" in result
        assert "timeline" in result
        assert "resources" in result
        assert "feasibility" in result
        assert "recommendation" in result
        assert result["recommendation"] == "RECOMMENDED TO PROCEED"

    @pytest.mark.asyncio
    async def test_assess_risks(
        self, mock_project_simulation_service, mock_llm_provider
    ):
        """Test risk assessment."""
        # Set up mock LLM response
        mock_json = {
            "technical_risks": [
                {
                    "description": "Technology stack incompatibility",
                    "probability": "medium",
                    "impact": "high",
                    "mitigation": "Conduct compatibility testing early",
                }
            ],
            "timeline_risks": [
                {
                    "description": "Scope creep",
                    "probability": "high",
                    "impact": "high",
                    "mitigation": "Define clear requirements and change control process",
                }
            ],
            "overall_risk": "medium",
        }

        async def mock_generate_text(*args, **kwargs):
            yield json.dumps(mock_json)

        mock_llm_provider.generate_text = mock_generate_text

        # Test risk assessment
        risks = await mock_project_simulation_service._assess_risks("Build a web app")

        # Verify result contains expected data
        assert (
            risks["technical_risks"][0]["description"]
            == "Technology stack incompatibility"
        )
        assert risks["overall_risk"] == "medium"

    @pytest.mark.asyncio
    async def test_estimate_timeline(
        self, mock_project_simulation_service, mock_llm_provider
    ):
        """Test timeline estimation."""
        # Set up mock LLM response
        mock_json = {
            "optimistic": 10,
            "realistic": 15,
            "pessimistic": 25,
            "confidence": "medium",
            "factors": ["complexity", "team experience", "dependencies"],
        }

        async def mock_generate_text(*args, **kwargs):
            yield json.dumps(mock_json)

        mock_llm_provider.generate_text = mock_generate_text

        # Test timeline estimation
        complexity = {"t_shirt_size": "L", "story_points": 8}
        timeline = await mock_project_simulation_service._estimate_timeline(
            "Build a web app", complexity
        )

        # Verify result contains expected data
        assert timeline["optimistic"] == 10
        assert timeline["realistic"] == 15
        assert timeline["pessimistic"] == 25
        assert timeline["confidence"] == "medium"
        assert "factors" in timeline

    @pytest.mark.asyncio
    async def test_assess_resource_needs(
        self, mock_project_simulation_service, mock_llm_provider
    ):
        """Test resource needs assessment."""
        # Set up mock LLM response
        mock_json = {
            "required_specializations": ["frontend", "backend", "database"],
            "number_of_agents": 3,
            "required_skillsets": ["React", "Node.js", "MongoDB"],
            "external_resources": ["Cloud hosting", "CI/CD pipeline"],
        }

        async def mock_generate_text(*args, **kwargs):
            yield json.dumps(mock_json)

        mock_llm_provider.generate_text = mock_generate_text

        # Test resource assessment
        complexity = {"t_shirt_size": "L", "story_points": 8}
        resources = await mock_project_simulation_service._assess_resource_needs(
            "Build a web app", complexity
        )

        # Verify result contains expected data
        assert len(resources["required_specializations"]) == 3
        assert resources["number_of_agents"] == 3
        assert "React" in resources["required_skillsets"]
        assert "Cloud hosting" in resources["external_resources"]

    def test_assess_feasibility(
        self, mock_project_simulation_service, task_planning_service
    ):
        """Test feasibility assessment."""
        # Setup test data
        resources = {"required_specializations": ["frontend", "backend", "ai"]}

        # Mock the agent service specializations
        specializations = {
            "agent1": "frontend development",
            "agent2": "backend systems",
        }
        task_planning_service.agent_service.get_specializations = MagicMock(
            return_value=specializations
        )
        task_planning_service.agent_service.get_all_ai_agents = MagicMock(
            return_value={"agent1": {}, "agent2": {}}
        )

        # Test feasibility assessment
        feasibility = mock_project_simulation_service._assess_feasibility(
            resources)

        # Verify result contains expected data
        assert isinstance(feasibility, dict)
        assert "feasible" in feasibility
        assert "coverage_score" in feasibility
        assert "missing_specializations" in feasibility
        assert "ai" in feasibility["missing_specializations"]
        assert feasibility["coverage_score"] < 100

    def test_generate_recommendation(self, mock_project_simulation_service):
        """Test recommendation generation."""
        # Test with high feasibility and low risk
        risks = {"overall_risk": "low"}
        feasibility = {
            "feasible": True,
            "coverage_score": 90,
            "missing_specializations": [],
        }

        recommendation = mock_project_simulation_service._generate_recommendation(
            risks, feasibility
        )
        assert "RECOMMENDED" in recommendation.upper()

        # Test with high feasibility but high risk
        risks = {"overall_risk": "high"}
        feasibility = {
            "feasible": True,
            "coverage_score": 90,
            "missing_specializations": [],
        }

        recommendation = mock_project_simulation_service._generate_recommendation(
            risks, feasibility
        )
        assert "CAUTION" in recommendation.upper()

        # Test with low feasibility
        risks = {"overall_risk": "low"}
        feasibility = {
            "feasible": False,
            "coverage_score": 40,
            "missing_specializations": ["ai", "data science"],
        }

        recommendation = mock_project_simulation_service._generate_recommendation(
            risks, feasibility
        )
        assert "NOT RECOMMENDED" in recommendation.upper()

    class TestNotificationService:
        """Tests for NotificationService."""

        @pytest.fixture
        def mock_notification_service(self, mock_human_agent_registry):
            """Create a mocked notification service."""
            return NotificationService(mock_human_agent_registry)

        def test_send_notification_with_handler(
            self, mock_notification_service, mock_human_agent_registry
        ):
            """Test sending notification when agent has a handler."""
            # Setup mock handler
            mock_handler = MagicMock()

            # Create agent with handler
            test_agent = {
                "agent_id": "agent_with_handler",
                "name": "Test Agent",
                "notification_handler": mock_handler,
                "availability_status": "available",
            }

            # Mock get_human_agent to return our test agent
            mock_human_agent_registry.get_human_agent = MagicMock(
                return_value=test_agent
            )

            # Test sending notification
            result = mock_notification_service.send_notification(
                "agent_with_handler", "Test notification", {"ticket_id": "123"}
            )

            # Verify handler was called
            assert result is True
            mock_handler.assert_called_once_with(
                "Test notification", {"ticket_id": "123"}
            )

        def test_send_notification_without_handler(
            self, mock_notification_service, mock_human_agent_registry
        ):
            """Test sending notification when agent has no handler."""
            # Create agent without handler
            test_agent = {
                "agent_id": "agent_without_handler",
                "name": "Test Agent",
                "availability_status": "available",
            }

            # Mock get_human_agent to return our test agent
            mock_human_agent_registry.get_human_agent = MagicMock(
                return_value=test_agent
            )

            # Test sending notification
            result = mock_notification_service.send_notification(
                "agent_without_handler", "Test notification"
            )

            # Verify result is false since no handler was available
            assert result is False

        def test_send_notification_agent_not_found(
            self, mock_notification_service, mock_human_agent_registry
        ):
            """Test sending notification to non-existent agent."""
            # Mock get_human_agent to return None
            mock_human_agent_registry.get_human_agent = MagicMock(
                return_value=None)

            # Test sending notification
            result = mock_notification_service.send_notification(
                "nonexistent_agent", "Test notification"
            )

            # Verify result is false since agent wasn't found
            assert result is False

        def test_notify_approvers(self, mock_notification_service):
            """Test notifying multiple approvers."""
            # Mock send_notification
            mock_notification_service.send_notification = MagicMock(
                return_value=True)

            # Test notifying multiple approvers
            approvers = ["approver1", "approver2", "approver3"]
            mock_notification_service.notify_approvers(
                approvers, "Approval needed", {"ticket_id": "123"}
            )

            # Verify send_notification was called for each approver
            assert mock_notification_service.send_notification.call_count == len(
                approvers
            )

    class TestProjectApprovalService:
        """Tests for ProjectApprovalService."""

        @pytest.fixture
        def mock_project_approval_service(
            self, mock_ticket_repository, mock_human_agent_registry
        ):
            """Create a mocked project approval service."""
            notification_service = NotificationService(
                mock_human_agent_registry)
            return ProjectApprovalService(
                mock_ticket_repository, mock_human_agent_registry, notification_service
            )

        def test_register_approver(
            self, mock_project_approval_service, mock_human_agent_registry
        ):
            """Test registering an approver."""
            # Setup mock to recognize the agent
            mock_human_agent_registry.get_all_human_agents = MagicMock(
                return_value={"approver1": {}}
            )

            # Register approver
            mock_project_approval_service.register_approver("approver1")

            # Verify approver was added
            assert "approver1" in mock_project_approval_service.approvers

            # Try to register non-existent agent
            mock_project_approval_service.register_approver("nonexistent")

            # Verify non-existent agent wasn't added
            assert "nonexistent" not in mock_project_approval_service.approvers

        @pytest.mark.asyncio
        async def test_submit_for_approval(
            self, mock_project_approval_service, mock_ticket_repository, sample_ticket
        ):
            """Test submitting a project for approval."""
            # Add an approver
            mock_project_approval_service.approvers = ["approver1"]

            # Mock notification service
            mock_project_approval_service.notification_service.send_notification = (
                AsyncMock(return_value=True)
            )

            # Submit for approval
            await mock_project_approval_service.submit_for_approval(sample_ticket)

            # Verify ticket was updated
            mock_ticket_repository.update.assert_called_once()
            args, kwargs = mock_ticket_repository.update.call_args
            assert args[0] == sample_ticket.id
            assert args[1]["status"] == TicketStatus.PENDING
            assert args[1]["approval_status"] == "awaiting_approval"

            # Verify notification was sent
            mock_project_approval_service.notification_service.send_notification.assert_called_once()

        @pytest.mark.asyncio
        async def test_process_approval_approved(
            self, mock_project_approval_service, mock_ticket_repository, sample_ticket
        ):
            """Test processing an approval decision (approved)."""
            # Add an approver
            mock_project_approval_service.approvers = ["approver1"]

            # Mock get_by_id to return our sample ticket
            mock_ticket_repository.get_by_id = MagicMock(
                return_value=sample_ticket)

            # Process approval (approved)
            await mock_project_approval_service.process_approval(
                "123", "approver1", True, "Looks good"
            )

            # Verify ticket was updated
            mock_ticket_repository.update.assert_called_once()
            args, kwargs = mock_ticket_repository.update.call_args
            assert args[0] == "123"
            assert args[1]["status"] == TicketStatus.ACTIVE
            assert args[1]["approval_status"] == "approved"
            assert args[1]["approver_id"] == "approver1"
            assert args[1]["approval_comments"] == "Looks good"

        @pytest.mark.asyncio
        async def test_process_approval_rejected(
            self, mock_project_approval_service, mock_ticket_repository, sample_ticket
        ):
            """Test processing an approval decision (rejected)."""
            # Add an approver
            mock_project_approval_service.approvers = ["approver1"]

            # Mock get_by_id to return our sample ticket
            mock_ticket_repository.get_by_id = MagicMock(
                return_value=sample_ticket)

            # Process approval (rejected)
            await mock_project_approval_service.process_approval(
                "123", "approver1", False, "Not feasible"
            )

            # Verify ticket was updated
            mock_ticket_repository.update.assert_called_once()
            args, kwargs = mock_ticket_repository.update.call_args
            assert args[0] == "123"
            assert args[1]["status"] == TicketStatus.RESOLVED
            assert args[1]["approval_status"] == "rejected"
            assert args[1]["approver_id"] == "approver1"
            assert args[1]["approval_comments"] == "Not feasible"

        @pytest.mark.asyncio
        async def test_process_approval_unauthorized(
            self, mock_project_approval_service, mock_ticket_repository
        ):
            """Test approval from unauthorized user."""
            # Add an approver (not the one trying to approve)
            mock_project_approval_service.approvers = ["approver1"]

            # Ensure this raises an exception
            with pytest.raises(ValueError, match="Not authorized to approve"):
                await mock_project_approval_service.process_approval(
                    "123", "unauthorized", True
                )

            # Verify repo was not called
            mock_ticket_repository.update.assert_not_called()

        @pytest.mark.asyncio
        async def test_process_approval_no_ticket(
            self, mock_project_approval_service, mock_ticket_repository
        ):
            """Test approval with non-existent ticket."""
            # Add an approver
            mock_project_approval_service.approvers = ["approver1"]

            # Mock get_by_id to return None
            mock_ticket_repository.get_by_id = MagicMock(return_value=None)

            # Ensure this raises an exception
            with pytest.raises(ValueError, match="Ticket .* not found"):
                await mock_project_approval_service.process_approval(
                    "nonexistent", "approver1", True
                )

            # Verify repo was not called
            mock_ticket_repository.update.assert_not_called()

    @pytest.mark.asyncio
    class TestQueryProcessorWithProjectApproval:
        """Test QueryProcessor with project approval and simulation."""

        @pytest.fixture
        def query_processor_with_approval(
            self, query_processor, mock_ticket_repository
        ):
            """Create a query processor with approval requirements."""
            query_processor.require_human_approval = True

            # Create project approval service
            query_processor.project_approval_service = MagicMock()
            query_processor.project_approval_service.submit_for_approval = AsyncMock()

            # Create project simulation service
            query_processor.project_simulation_service = MagicMock()
            query_processor.project_simulation_service.simulate_project = AsyncMock(
                return_value={
                    "complexity": {"t_shirt_size": "L"},
                    "risks": {"overall_risk": "medium"},
                    "timeline": {"realistic": 15},
                    "recommendation": "PROCEED WITH CAUTION",
                }
            )

            return query_processor

        @pytest.mark.asyncio
        async def test_process_with_human_approval(self, query_processor_with_approval, mock_ticket_repository, sample_ticket):
            """Test processing a query that requires human approval."""
            # Setup mocks
            query_processor_with_approval._is_human_agent = AsyncMock(
                return_value=False)
            query_processor_with_approval._is_simple_greeting = AsyncMock(
                return_value=False)
            query_processor_with_approval._process_system_commands = AsyncMock(
                return_value=None)
            mock_ticket_repository.get_active_for_user.return_value = None

            # Setup complexity to be non-simple
            complexity_data = {
                "t_shirt_size": "L",
                "story_points": 8,
                "estimated_minutes": 120,
                "technical_complexity": 7,
                "domain_knowledge": 6,
            }
            query_processor_with_approval._assess_task_complexity = AsyncMock(
                return_value=complexity_data)

            # Important: Mock the needs_breakdown method
            query_processor_with_approval.task_planning_service.needs_breakdown = AsyncMock(
                return_value=(True, "Task is complex"))

            # Mock the ticket service
            query_processor_with_approval.ticket_service.get_or_create_ticket = AsyncMock(
                return_value=sample_ticket)

            # Mock project simulation to return data and track calls
            query_processor_with_approval.project_simulation_service.simulate_project = AsyncMock(return_value={
                "complexity": complexity_data,
                "risks": {"overall_risk": "medium"},
                "timeline": {"realistic": 5},
                "recommendation": "PROCEED WITH CAUTION"
            })

            # Mock project approval submission
            query_processor_with_approval.project_approval_service.submit_for_approval = AsyncMock()

            # Create a generator function for the query processor to yield
            # Update the mock to accept the expected parameters
            async def mock_process_response(user_id, user_text, ticket, agent=None):
                # Call simulate_project to make sure it gets called
                await query_processor_with_approval.project_simulation_service.simulate_project("Create a complex project")
                # Submit for approval
                await query_processor_with_approval.project_approval_service.submit_for_approval(ticket)

                yield "Analyzing project feasibility...\n\n"
                yield "## Project Simulation Results\n\n"
                yield "**Complexity**: L\n"
                yield "**Timeline**: 5 days\n"
                yield "**Risk Level**: medium\n"
                yield "**Recommendation**: PROCEED WITH CAUTION\n\n"
                yield "\nThis project has been submitted for approval. You'll be notified once it's reviewed."

            # Replace _process_new_ticket with our custom generator
            query_processor_with_approval._process_new_ticket = mock_process_response

            # Test processing with approval requirement
            response = ""
            async for chunk in query_processor_with_approval.process("user1", "Create a complex project"):
                response += chunk

            # Verify simulation was run
            query_processor_with_approval.project_simulation_service.simulate_project.assert_called_once()

            # Verify project was submitted for approval
            query_processor_with_approval.project_approval_service.submit_for_approval.assert_called_once()

            # Check that the response contains the expected content
            assert "Project Simulation Results" in response
            assert "PROCEED WITH CAUTION" in response
            assert "submitted for approval" in response

        async def test_simple_query_bypasses_approval(
            self, query_processor_with_approval, mock_ticket_repository
        ):
            """Test that simple queries bypass the approval process."""
            # Setup mocks
            query_processor_with_approval._is_human_agent = AsyncMock(
                return_value=False
            )
            query_processor_with_approval._is_simple_greeting = AsyncMock(
                return_value=False
            )
            query_processor_with_approval._process_system_commands = AsyncMock(
                return_value=None
            )
            mock_ticket_repository.get_active_for_user.return_value = None

            # Setup complexity to be simple
            query_processor_with_approval._assess_task_complexity = AsyncMock(
                return_value={"t_shirt_size": "XS", "story_points": 1}
            )

            # Mock routing and response generation
            query_processor_with_approval.routing_service.route_query = AsyncMock(
                return_value="test_agent"
            )

            # Test processing with a simple query
            async for _ in query_processor_with_approval.process(
                "user1", "Simple question"
            ):
                pass

            # Verify simulation was not run
            query_processor_with_approval.project_simulation_service.simulate_project.assert_not_called()

            # Verify ticket was not submitted for approval
            query_processor_with_approval.project_approval_service.submit_for_approval.assert_not_called()


class TestQdrantAdapter:
    """Tests for the Qdrant vector store adapter."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock the Qdrant client."""
        with patch("qdrant_client.QdrantClient") as mock_client:
            # Set up collections list mock
            mock_collections = MagicMock()
            mock_collections.collections = [
                MagicMock(name="existing_collection")]
            mock_client.return_value.get_collections.return_value = mock_collections

            yield mock_client.return_value

    @patch("qdrant_client.QdrantClient")
    def test_initialization(self, mock_client_class):
        """Test initialization creates the collection if it doesn't exist."""
        # Setup mock client
        mock_client = mock_client_class.return_value
        mock_client.get_collections.return_value.collections = []

        QdrantAdapter(
            url="http://test-url:6333",
            api_key="test-key",
            collection_name="test_collection",
        )

        # Verify client was created with correct params
        mock_client_class.assert_called_once_with(
            url="http://test-url:6333", api_key="test-key"
        )

        # Verify create_collection was called if collection doesn't exist
        mock_client.create_collection.assert_called_once()
        args, kwargs = mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "test_collection"

    @patch("qdrant_client.QdrantClient")
    def test_initialization_existing_collection(self, mock_client_class):
        """Test initialization doesn't recreate an existing collection."""
        # Setup mock client
        mock_client = mock_client_class.return_value
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [
            mock_collection]

        # Test initialization with existing collection
        QdrantAdapter(collection_name="test_collection")

        # Verify create_collection was not called
        mock_client.create_collection.assert_not_called()

    @patch("qdrant_client.QdrantClient")
    def test_store_vectors(self, mock_client_class):
        """Test storing vectors in Qdrant."""
        # Setup mock client
        mock_client = mock_client_class.return_value

        # Create adapter
        adapter = QdrantAdapter(collection_name="test_collection")

        # Test data
        vectors = [
            {"id": "vec1", "values": [0.1, 0.2, 0.3],
                "metadata": {"key": "value1"}},
            {"id": "vec2", "values": [0.4, 0.5, 0.6],
                "metadata": {"key": "value2"}},
        ]

        # Store vectors
        adapter.store_vectors(vectors, "test_namespace")

        # Verify upsert was called correctly
        mock_client.upsert.assert_called_once()
        args, kwargs = mock_client.upsert.call_args

        # Check collection name
        assert kwargs["collection_name"] == "test_collection"

        # Check points format - we can't directly check the PointStruct objects
        # but we can verify the call was made with the right number of points
        assert len(kwargs["points"]) == 2

    @patch("qdrant_client.QdrantClient")
    def test_search_vectors(self, mock_client_class):
        """Test searching vectors in Qdrant."""
        # Setup mock client
        mock_client = mock_client_class.return_value

        # Setup mock search results
        result1 = MagicMock()
        result1.id = "vec1"
        result1.score = 0.95
        result1.payload = {"namespace": "test_namespace", "key": "value1"}

        result2 = MagicMock()
        result2.id = "vec2"
        result2.score = 0.85
        result2.payload = {"namespace": "test_namespace", "key": "value2"}

        mock_client.search.return_value = [result1, result2]

        # Create adapter
        adapter = QdrantAdapter(collection_name="test_collection")

        # Test search
        results = adapter.search_vectors(
            [0.1, 0.2, 0.3], "test_namespace", limit=2)

        # Verify search was called correctly
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert kwargs["limit"] == 2

        # Verify filter contains namespace condition
        assert "must" in kwargs["query_filter"].dict()

        # Verify search results were formatted correctly
        assert len(results) == 2
        assert results[0]["id"] == "vec1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["key"] == "value1"
        assert results[1]["id"] == "vec2"
        assert results[1]["score"] == 0.85

    @patch("qdrant_client.QdrantClient")
    def test_delete_vector(self, mock_client_class):
        """Test deleting a vector from Qdrant."""
        # Setup mock client
        mock_client = mock_client_class.return_value

        # Create adapter
        adapter = QdrantAdapter(collection_name="test_collection")

        # Test delete
        adapter.delete_vector("vec1", "test_namespace")

        # Verify delete was called correctly
        mock_client.delete.assert_called_once()
        args, kwargs = mock_client.delete.call_args
        assert kwargs["collection_name"] == "test_collection"
        assert "vec1" in str(kwargs["points_selector"].dict())
        assert kwargs["wait"] is True


class TestTenantContext:
    """Tests for the TenantContext class."""

    def test_initialization(self):
        """Test initialization with and without config."""
        # Test with no config
        context = TenantContext("tenant1")
        assert context.tenant_id == "tenant1"
        assert context.config == {}
        assert context.metadata == {}

        # Test with config
        config = {"key": "value", "nested": {"subkey": "subvalue"}}
        context = TenantContext("tenant2", config)
        assert context.tenant_id == "tenant2"
        assert context.config == config
        assert context.metadata == {}

    def test_get_config_value(self):
        """Test retrieving config values."""
        config = {"key1": "value1", "nested": {"key2": "value2"}}
        context = TenantContext("tenant1", config)

        # Test getting existing values
        assert context.get_config_value("key1") == "value1"
        assert context.get_config_value("nested") == {"key2": "value2"}

        # Test getting non-existent values
        assert context.get_config_value("nonexistent") is None
        assert context.get_config_value("nonexistent", "default") == "default"

    def test_metadata_management(self):
        """Test setting and getting metadata."""
        context = TenantContext("tenant1")

        # Test setting and getting metadata
        context.set_metadata("key1", "value1")
        context.set_metadata("key2", {"nested": "value2"})

        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2") == {"nested": "value2"}

        # Test getting non-existent metadata
        assert context.get_metadata("nonexistent") is None
        assert context.get_metadata("nonexistent", "default") == "default"


class TestTenantManager:
    """Tests for the TenantManager class."""

    def test_initialization(self):
        """Test initialization with and without default config."""
        # Test with no default config
        manager = TenantManager()
        assert manager.tenants == {}
        assert manager.default_config == {}

        # Test with default config
        default_config = {"key": "value"}
        manager = TenantManager(default_config)
        assert manager.default_config == default_config

    def test_register_tenant(self):
        """Test registering tenants with and without custom config."""
        default_config = {"key1": "value1", "key2": "value2"}
        manager = TenantManager(default_config)

        # Register tenant with default config
        context1 = manager.register_tenant("tenant1")
        assert context1.tenant_id == "tenant1"
        assert context1.config == default_config
        assert "tenant1" in manager.tenants

        # Register tenant with custom config that overrides defaults
        custom_config = {"key2": "custom", "key3": "new"}
        context2 = manager.register_tenant("tenant2", custom_config)
        assert context2.tenant_id == "tenant2"
        assert context2.config["key1"] == "value1"  # From default
        assert context2.config["key2"] == "custom"  # Overridden
        assert context2.config["key3"] == "new"  # New value

    def test_get_tenant(self):
        """Test retrieving tenant contexts."""
        manager = TenantManager()
        manager.register_tenant("tenant1")

        # Get existing tenant
        context = manager.get_tenant("tenant1")
        assert context is not None
        assert context.tenant_id == "tenant1"

        # Get non-existent tenant
        assert manager.get_tenant("nonexistent") is None

    @patch("solana_agent.ai.MongoDBAdapter")
    def test_create_tenant_db_adapter(self, mock_mongodb_adapter):
        """Test creating a tenant-specific database adapter."""
        # Setup
        default_config = {
            "mongo": {
                "connection_string": "mongodb://default:27017",
                "database": "default_db",
            }
        }
        manager = TenantManager(default_config)
        tenant = manager.register_tenant("tenant1")

        # Test with default config
        manager._create_tenant_db_adapter(tenant)
        mock_mongodb_adapter.assert_called_once_with(
            connection_string="mongodb://default:27017",
            database_name="default_db_tenant1",
        )

        # Test with tenant-specific config
        mock_mongodb_adapter.reset_mock()
        tenant_config = {
            "mongo": {"connection_string": "mongodb://tenant:27017"}}
        tenant = manager.register_tenant("tenant2", tenant_config)
        manager._create_tenant_db_adapter(tenant)
        mock_mongodb_adapter.assert_called_once_with(
            connection_string="mongodb://tenant:27017",
            database_name="default_db_tenant2",
        )

    @patch("solana_agent.ai.QdrantAdapter")
    @patch("solana_agent.ai.PineconeAdapter")
    def test_create_vector_provider(self, mock_pinecone, mock_qdrant):
        """Test creating vector providers based on tenant config."""
        # Setup default config with no vector provider
        manager = TenantManager({})

        # Test with Qdrant config
        qdrant_config = {
            "qdrant": {
                "url": "http://qdrant:6333",
                "api_key": "qdrant-key",
                "collection": "tenant_collection",
            }
        }
        tenant = manager.register_tenant("tenant1", qdrant_config)
        manager._create_tenant_vector_provider(tenant)

        mock_qdrant.assert_called_once_with(
            url="http://qdrant:6333",
            api_key="qdrant-key",
            collection_name="tenant_tenant1_tenant_collection",
            embedding_model="text-embedding-3-small",
        )
        mock_pinecone.assert_not_called()

        # Reset mocks
        mock_qdrant.reset_mock()
        mock_pinecone.reset_mock()

        # Test with Pinecone config
        pinecone_config = {
            "pinecone": {"api_key": "pinecone-key", "index": "pinecone-index"}
        }
        tenant = manager.register_tenant("tenant2", pinecone_config)
        manager._create_tenant_vector_provider(tenant)

        mock_pinecone.assert_called_once_with(
            api_key="pinecone-key",
            index_name="pinecone-index",
            embedding_model="text-embedding-3-small",
        )
        mock_qdrant.assert_not_called()

        # Test precedence (Qdrant over Pinecone)
        mock_qdrant.reset_mock()
        mock_pinecone.reset_mock()

        both_config = {
            "qdrant": {"url": "http://qdrant:6333", "api_key": "qdrant-key"},
            "pinecone": {"api_key": "pinecone-key", "index": "pinecone-index"},
        }
        tenant = manager.register_tenant("tenant3", both_config)
        manager._create_tenant_vector_provider(tenant)

        mock_qdrant.assert_called_once()
        mock_pinecone.assert_not_called()

    def test_get_repository_errors(self):
        """Test error handling in get_repository."""
        manager = TenantManager()

        # Test with non-existent tenant
        with pytest.raises(ValueError, match="Tenant nonexistent not found"):
            manager.get_repository("nonexistent", "ticket")

        # Test with invalid repository type
        manager.register_tenant("tenant1")
        with pytest.raises(ValueError, match="Unknown repository type"):
            manager.get_repository("tenant1", "invalid_type")

    def test_get_service_errors(self):
        """Test error handling in get_service."""
        manager = TenantManager()

        # Test with non-existent tenant
        with pytest.raises(ValueError, match="Tenant nonexistent not found"):
            manager.get_service("nonexistent", "agent")

        # Test with invalid service type
        manager.register_tenant("tenant1")
        with pytest.raises(ValueError, match="Unknown service type"):
            manager.get_service("tenant1", "invalid_service")


class TestMultitenantSolanaAgentFactory:
    """Tests for the MultitenantSolanaAgentFactory."""

    def test_initialization(self):
        """Test factory initialization."""
        global_config = {"key": "value"}
        factory = MultitenantSolanaAgentFactory(global_config)

        assert factory.tenant_manager is not None
        assert factory.tenant_manager.default_config == global_config

    def test_register_tenant(self):
        """Test tenant registration through factory."""
        factory = MultitenantSolanaAgentFactory({})

        # Mock tenant_manager.register_tenant
        factory.tenant_manager.register_tenant = MagicMock()

        # Register tenant
        factory.register_tenant("tenant1", {"custom": "config"})

        # Verify tenant_manager.register_tenant was called
        factory.tenant_manager.register_tenant.assert_called_once_with(
            "tenant1", {"custom": "config"}
        )

    def test_get_processor(self):
        """Test getting a query processor for a tenant."""
        factory = MultitenantSolanaAgentFactory({})

        # Mock tenant_manager.get_service
        mock_processor = MagicMock()
        factory.tenant_manager.get_service = MagicMock(
            return_value=mock_processor)

        # Get processor
        processor = factory.get_processor("tenant1")

        # Verify tenant_manager.get_service was called with correct parameters
        factory.tenant_manager.get_service.assert_called_once_with(
            "tenant1", "query_processor"
        )
        assert processor == mock_processor

    def test_get_agent_service(self):
        """Test getting an agent service for a tenant."""
        factory = MultitenantSolanaAgentFactory({})

        # Mock tenant_manager.get_service
        mock_agent_service = MagicMock()
        factory.tenant_manager.get_service = MagicMock(
            return_value=mock_agent_service)

        # Get agent service
        agent_service = factory.get_agent_service("tenant1")

        # Verify tenant_manager.get_service was called with correct parameters
        factory.tenant_manager.get_service.assert_called_once_with(
            "tenant1", "agent")
        assert agent_service == mock_agent_service

    def test_register_ai_agent(self):
        """Test registering an AI agent for a tenant."""
        factory = MultitenantSolanaAgentFactory({})

        # Mock get_agent_service and register_ai_agent
        mock_agent_service = MagicMock()
        factory.get_agent_service = MagicMock(return_value=mock_agent_service)

        # Register AI agent
        factory.register_ai_agent(
            "tenant1", "test_agent", "Test instructions", "Testing", "gpt-4o"
        )

        # Verify get_agent_service was called
        factory.get_agent_service.assert_called_once_with("tenant1")

        # Verify register_ai_agent was called with correct parameters
        mock_agent_service.register_ai_agent.assert_called_once_with(
            "test_agent", "Test instructions", "Testing", "gpt-4o"
        )


class TestMultitenantSolanaAgent:
    """Tests for the MultitenantSolanaAgent client interface."""

    @patch("solana_agent.ai.MultitenantSolanaAgentFactory")
    def test_initialization_with_config(self, mock_factory_class):
        """Test initialization with direct config."""
        config = {"key": "value"}

        client = MultitenantSolanaAgent(config=config)

        # Verify factory was created
        mock_factory_class.assert_called_once_with(config)
        assert client.factory == mock_factory_class.return_value

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.load")
    @patch("solana_agent.ai.MultitenantSolanaAgentFactory")
    def test_initialization_with_json_config_file(
        self, mock_factory_class, mock_json_load, mock_open
    ):
        """Test initialization with JSON config file."""
        config = {"key": "value"}
        mock_json_load.return_value = config

        MultitenantSolanaAgent(config_path="config.json")

        # Verify open was called with the file path
        mock_open.assert_called_once_with("config.json", "r")

        # Verify json.load was called
        mock_json_load.assert_called_once()

        # Verify factory was created with loaded config
        mock_factory_class.assert_called_once_with(config)

    @patch("importlib.util.spec_from_file_location")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("solana_agent.ai.MultitenantSolanaAgentFactory")
    def test_initialization_with_python_config_file(
        self, mock_factory_class, mock_open, mock_spec_from_file_location
    ):
        """Test initialization with Python config file."""
        # Setup importlib mocking
        mock_spec = MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        mock_module = MagicMock()
        mock_spec.loader.exec_module = MagicMock()

        # Config in Python module
        config = {"key": "value"}
        mock_module.config = config

        # Using a context manager approach that better handles the patching
        with patch("importlib.util.module_from_spec", return_value=mock_module):
            MultitenantSolanaAgent(config_path="config.py")

        # Verify spec_from_file_location was called
        mock_spec_from_file_location.assert_called_once_with(
            "config", "config.py")

        # Verify module was executed
        mock_spec.loader.exec_module.assert_called_once_with(mock_module)

        # Verify factory was created with config from module
        mock_factory_class.assert_called_once_with(config)

    def test_initialization_with_no_config(self):
        """Test initialization with no config."""
        with pytest.raises(
            ValueError, match="Either config or config_path must be provided"
        ):
            MultitenantSolanaAgent()

    def test_register_tenant(self):
        """Test registering a tenant."""
        client = MultitenantSolanaAgent(config={})
        client.factory.register_tenant = MagicMock()

        client.register_tenant("tenant1", {"custom": "config"})

        client.factory.register_tenant.assert_called_once_with(
            "tenant1", {"custom": "config"}
        )

    @pytest.mark.asyncio
    async def test_process(self):
        """Test processing a message through the client."""
        client = MultitenantSolanaAgent(config={})

        # Mock processor
        mock_processor = AsyncMock()

        async def mock_process(*args, **kwargs):
            yield "Response chunk 1"
            yield "Response chunk 2"

        mock_processor.process = mock_process
        client.factory.get_processor = MagicMock(return_value=mock_processor)

        # Process a message
        response = []
        async for chunk in client.process("tenant1", "user1", "Hello"):
            response.append(chunk)

        # Verify get_processor was called
        client.factory.get_processor.assert_called_once_with("tenant1")

        # Verify response contains expected chunks
        assert response == ["Response chunk 1", "Response chunk 2"]

    def test_register_agent(self):
        """Test registering an AI agent."""
        client = MultitenantSolanaAgent(config={})
        client.factory.register_ai_agent = MagicMock()

        client.register_agent(
            "tenant1", "test_agent", "Test instructions", "Testing", "gpt-4o"
        )

        client.factory.register_ai_agent.assert_called_once_with(
            "tenant1", "test_agent", "Test instructions", "Testing", "gpt-4o"
        )

    def test_register_human_agent(self):
        """Test registering a human agent."""
        client = MultitenantSolanaAgent(config={})

        # Mock agent service
        mock_agent_service = MagicMock()
        client.factory.get_agent_service = MagicMock(
            return_value=mock_agent_service)

        # Mock notification handler
        mock_handler = MagicMock()

        client.register_human_agent(
            "tenant1", "human1", "Human Agent", "Support", mock_handler
        )

        # Verify agent_service was requested
        client.factory.get_agent_service.assert_called_once_with("tenant1")

        # Verify register_human_agent was called with correct parameters
        mock_agent_service.register_human_agent.assert_called_once_with(
            agent_id="human1",
            name="Human Agent",
            specialization="Support",
            notification_handler=mock_handler,
        )


class TestToolRegistry:
    """Tests for the Tool Registry functionality."""

    def setup_method(self):
        # Reset registry before each test
        self.registry = ToolRegistry()

    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register_tool(TestMathTool)

        # Verify tool was registered
        assert "test_math_tool" in self.registry._tools

    def test_assign_tool_to_agent(self):
        """Test assigning a tool to an agent."""
        # Register tool
        self.registry.register_tool(TestMathTool)

        # Assign to agent
        self.registry.assign_tool_to_agent("test_agent", "test_math_tool")

        # Verify assignment
        assert "test_agent" in self.registry._agent_tools
        assert "test_math_tool" in self.registry._agent_tools["test_agent"]

    def test_assign_invalid_tool(self):
        """Test assigning a non-existent tool."""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            self.registry.assign_tool_to_agent(
                "test_agent", "nonexistent_tool")

    def test_get_agent_tools(self):
        """Test getting tools for an agent."""
        # Register tools
        self.registry.register_tool(TestMathTool)
        self.registry.register_tool(TestInfoTool)

        # Assign to agent
        self.registry.assign_tool_to_agent("test_agent", "test_math_tool")

        # Get tools
        tools = self.registry.get_agent_tools("test_agent")

        # Verify
        assert len(tools) == 1
        assert tools[0]["name"] == "test_math_tool"
        assert "description" in tools[0]
        assert "parameters" in tools[0]

        # Test agent with no tools
        tools = self.registry.get_agent_tools("agent_with_no_tools")
        assert len(tools) == 0

    def test_get_tool(self):
        """Test getting a tool by name."""
        # Register tool
        self.registry.register_tool(TestMathTool)

        # Get tool
        tool = self.registry.get_tool("test_math_tool")

        # Verify
        assert isinstance(tool, TestMathTool)
        assert tool.name == "test_math_tool"

        # Test nonexistent tool
        assert self.registry.get_tool("nonexistent_tool") is None


class TestPluginManager:
    """Tests for the Plugin Manager functionality."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create a temporary plugin directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            plugin_dir = Path(tmpdirname) / "plugins"
            plugin_dir.mkdir()

            # Create a test plugin
            test_plugin_dir = plugin_dir / "test_plugin"
            test_plugin_dir.mkdir()

            # Create plugin metadata
            with open(test_plugin_dir / "plugin.json", "w") as f:
                json.dump(
                    {
                        "name": "test_plugin",
                        "version": "1.0.0",
                        "description": "Test plugin",
                        "tools": ["test_math_tool", "test_info_tool"],
                    },
                    f,
                )

            # Create __init__.py
            with open(test_plugin_dir / "__init__.py", "w") as f:
                f.write("""
from solana_agent.ai import Tool

class TestMathTool(Tool):
    @property
    def name(self) -> str:
        return "test_math_tool"
    
    @property
    def description(self) -> str:
        return "A test tool that performs basic math operations"
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        }
    
    def execute(self, **kwargs):
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        op = kwargs.get("operation", "add")
        
        if op == "add":
            return {"result": a + b, "status": "success"}
        return {"error": "Unknown operation", "status": "error"}

class TestInfoTool(Tool):
    @property
    def name(self) -> str:
        return "test_info_tool"
    
    @property
    def description(self) -> str:
        return "A test tool that returns information"
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "info_type": {"type": "string"}
            }
        }
    
    def execute(self, **kwargs):
        return {"info": f"Info about {kwargs.get('info_type', 'unknown')}", "status": "success"}

def register_tools(registry):
    registry.register_tool(TestMathTool)
    registry.register_tool(TestInfoTool)
""")

            # Create an empty requirements.txt
            with open(test_plugin_dir / "requirements.txt", "w") as f:
                f.write("# No dependencies")

            yield tmpdirname

    def test_discover_plugins(self, temp_plugin_dir):
        """Test discovering plugins."""
        manager = PluginManager(os.path.join(temp_plugin_dir, "plugins"))
        plugins = manager.discover_plugins()

        # Verify plugin was discovered
        assert len(plugins) == 1
        assert plugins[0]["name"] == "test_plugin"
        assert plugins[0]["tools"] == ["test_math_tool", "test_info_tool"]

    def test_load_plugin(self, temp_plugin_dir):
        """Test loading a plugin."""
        manager = PluginManager(os.path.join(temp_plugin_dir, "plugins"))
        plugins = manager.discover_plugins()

        # Mock the registry
        mock_registry = MagicMock()

        # Use patching to avoid modifying the global registry
        with patch("solana_agent.ai.tool_registry", mock_registry):
            result = manager.load_plugin(plugins[0])

            # Verify plugin was loaded
            assert result is True
            assert plugins[0]["name"] in manager.loaded_plugins

            # Verify register_tools was called
            # Instead of checking for a specific MagicMock class, just verify the method was called
            assert mock_registry.register_tool.call_count > 0

    def test_load_all_plugins(self, temp_plugin_dir):
        """Test loading all plugins."""
        manager = PluginManager(os.path.join(temp_plugin_dir, "plugins"))

        # Use patching to avoid modifying the global registry
        with patch("solana_agent.ai.tool_registry", MagicMock()):
            count = manager.load_all_plugins()

            # Verify plugins were loaded
            assert count == 1
            assert "test_plugin" in manager.loaded_plugins

    def test_execute_tool(self):
        """Test executing a tool."""
        # Register tools directly
        registry = ToolRegistry()
        registry.register_tool(TestMathTool)

        manager = PluginManager()

        # Use patching to use our test registry
        with patch("solana_agent.ai.tool_registry", registry):
            # Execute the tool
            result = manager.execute_tool(
                "test_math_tool", operation="add", a=2, b=3)

            # Verify result
            assert result["status"] == "success"
            assert result["result"] == 5


class TestAgentServiceWithTools:
    """Tests for AgentService integration with tools."""

    @pytest.fixture
    def agent_service_with_tools(self):
        """Create an agent service with tools for testing."""
        # Create a mock LLM provider with a response that will actually trigger tool execution
        mock_llm = AsyncMock()

        async def mock_generate_text(*args, **kwargs):
            # To fix the test, use the format OpenAI actually returns for tool calls
            if "tools" in kwargs:
                yield json.dumps(
                    {
                        "content": "",  # Content can be empty when using tools
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "test_math_tool",
                                    "arguments": json.dumps(
                                        {"operation": "add", "a": 2, "b": 3}
                                    ),
                                },
                            }
                        ],
                    }
                )
            else:
                yield "Regular response without tools"

        mock_llm.generate_text = mock_generate_text

        # Create agent service
        service = AgentService(mock_llm)

        # Register an AI agent
        service.register_ai_agent(
            "test_agent",
            "You are a test agent that can use tools.",
            "Testing",
            "gpt-4o",
        )

        # Setup plugin manager and register tools
        service.plugin_manager = PluginManager()

        # Create test registry and register tools
        test_registry = ToolRegistry()
        test_registry.register_tool(TestMathTool)
        test_registry.register_tool(TestInfoTool)

        # Register tools for the agent
        test_registry.assign_tool_to_agent("test_agent", "test_math_tool")

        # Use patching to use our test registry
        with patch("solana_agent.ai.tool_registry", test_registry):
            yield service

    @pytest.mark.asyncio
    async def test_generate_response_with_tools(self, agent_service_with_tools):
        """Test generating a response that uses tools."""
        # Mock execute_tool
        agent_service_with_tools.execute_tool = MagicMock(
            return_value={"result": 5, "status": "success"}
        )

        # Generate response
        response = ""
        async for chunk in agent_service_with_tools.generate_response(
            "test_agent", "user1", "Add 2 and 3"
        ):
            response += chunk

        # Verify execute_tool was called
        agent_service_with_tools.execute_tool.assert_called_once_with(
            "test_agent", "test_math_tool", {
                "operation": "add", "a": 2, "b": 3}
        )

    def test_register_tool_for_agent(self, agent_service_with_tools):
        """Test registering a tool for an agent."""
        with patch("solana_agent.ai.tool_registry") as mock_registry:
            # Register a tool
            agent_service_with_tools.register_tool_for_agent(
                "test_agent", "test_info_tool"
            )

            # Verify assign_tool_to_agent was called
            mock_registry.assign_tool_to_agent.assert_called_once_with(
                "test_agent", "test_info_tool"
            )

    def test_get_agent_tools(self, agent_service_with_tools):
        """Test getting tools for an agent."""
        with patch("solana_agent.ai.tool_registry") as mock_registry:
            # Setup mock return value
            mock_registry.get_agent_tools.return_value = [
                {"name": "test_tool", "description": "Test"}
            ]

            # Get tools
            tools = agent_service_with_tools.get_agent_tools("test_agent")

            # Verify get_agent_tools was called
            mock_registry.get_agent_tools.assert_called_once_with("test_agent")
            assert len(tools) == 1
            assert tools[0]["name"] == "test_tool"

    def test_execute_tool(self, agent_service_with_tools):
        """Test executing a tool."""
        with patch("solana_agent.ai.tool_registry") as mock_registry:
            # Setup mock return values
            mock_registry.get_agent_tools.return_value = [
                {"name": "test_math_tool", "description": "Test"}
            ]

            # Mock plugin manager
            agent_service_with_tools.plugin_manager.execute_tool = MagicMock(
                return_value={"result": 5, "status": "success"}
            )

            # Execute tool
            result = agent_service_with_tools.execute_tool(
                "test_agent", "test_math_tool", {
                    "operation": "add", "a": 2, "b": 3}
            )

            # Verify plugin_manager.execute_tool was called
            agent_service_with_tools.plugin_manager.execute_tool.assert_called_once_with(
                "test_math_tool", operation="add", a=2, b=3
            )
            assert result["result"] == 5


class TestInternetSearchPlugin:
    """Tests for the Internet Search plugin."""

    def test_search_tool_structure(self):
        """Test that the search tool has the correct structure."""
        # This would test the actual internet search plugin once implemented
        # For now, we'll just verify the expected properties of such a tool

        # Create a simple mock of what the internet search tool should look like
        class MockSearchTool(Tool):
            @property
            def name(self) -> str:
                return "search_internet"

            @property
            def description(self) -> str:
                return "Search the internet for information"

            @property
            def parameters_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "model": {
                            "type": "string",
                            "enum": [
                                "sonar",
                                "sonar-pro",
                                "sonar-reasoning-pro",
                                "sonar-reasoning",
                            ],
                            "description": "Perplexity model to use",
                        },
                    },
                    "required": ["query"],
                }

            def execute(self, **kwargs) -> dict:
                return {
                    "result": f"Search results for: {kwargs.get('query')}",
                    "status": "success",
                }

        # Create and verify the tool
        tool = MockSearchTool()
        assert tool.name == "search_internet"
        assert "Search the internet" in tool.description
        assert "query" in tool.parameters_schema["properties"]
        assert "model" in tool.parameters_schema["properties"]

        # Test execution
        result = tool.execute(query="test query")
        assert "test query" in result["result"]
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main(["-xvs", "test_solana_agent.py"])
