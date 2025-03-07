import datetime
import json
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from solana_agent.ai import (
    # Domain Models
    MongoHumanAgentRegistry,
    NotificationService,
    ProjectApprovalService,
    ProjectSimulationService,
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
)


#############################################
# FIXTURES
#############################################


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
        # Setup mock to return a JSON response
        mock_response = (
            '{"insights": [{"fact": "Test fact", "relevance": "Test relevance"}]}'
        )

        # We need to patch the generate_text method to return our mock JSON
        async def mock_generate_text(*args, **kwargs):
            yield mock_response

        mock_llm_provider.generate_text = mock_generate_text

        # Test insight extraction
        insights = await memory_service.extract_insights(
            "user1", {"message": "Hi", "response": "Hello"}
        )

        assert len(insights) == 1
        assert insights[0].fact == "Test fact"
        assert insights[0].relevance == "Test relevance"

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
    async def test_assess_task_complexity(
        self, task_planning_service, mock_llm_provider
    ):
        """Test assessing task complexity."""
        # Create a mock JSON response for complexity
        mock_json = {
            "t_shirt_size": "L",
            "story_points": 8,
            "estimated_minutes": 45,
            "technical_complexity": 7,
        }

        # Reset the mock and set a proper return value
        mock_llm_provider.generate_text = AsyncMock()
        mock_llm_provider.generate_text.return_value = [json.dumps(mock_json)]

        # This is testing a private method, so we need to access it directly
        complexity = await task_planning_service._assess_task_complexity(
            "Build a complex application"
        )

        # Since we're using a mock, we should get the complexity object
        assert "t_shirt_size" in complexity
        assert "story_points" in complexity
        assert "estimated_minutes" in complexity

        # Verify the LLM was called
        assert mock_llm_provider.generate_text.called


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

    async def test_process_new_ticket(
        self, query_processor, mock_ticket_repository, agent_service
    ):
        """Test processing a message creating a new ticket."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        query_processor.routing_service.route_query = AsyncMock(
            return_value="test_agent"
        )
        query_processor._assess_task_complexity = AsyncMock(
            return_value={"t_shirt_size": "M"}
        )
        mock_ticket_repository.get_active_for_user.return_value = None

        # Mock the ticket service methods directly
        query_processor.ticket_service.update_ticket_status = MagicMock(
            return_value=True
        )

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
            return_value=new_ticket
        )

        # Test processing a new ticket
        response = ""
        async for chunk in query_processor.process("user1", "Help me with something"):
            response += chunk

        assert response == "This is a test response"

        # Verify methods were called
        query_processor._is_human_agent.assert_called_once()
        query_processor._is_simple_greeting.assert_called_once()
        query_processor._process_system_commands.assert_called_once()
        query_processor.routing_service.route_query.assert_called_once()
        query_processor._assess_task_complexity.assert_called_once()
        query_processor.ticket_service.get_or_create_ticket.assert_called_once()
        assert query_processor.ticket_service.update_ticket_status.called

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
            "agents": [
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

        async def test_process_with_human_approval(
            self, query_processor_with_approval, mock_ticket_repository, sample_ticket
        ):
            """Test processing a query that requires human approval."""
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

            # Setup complexity to be non-simple
            query_processor_with_approval._assess_task_complexity = AsyncMock(
                return_value={"t_shirt_size": "L", "story_points": 8}
            )

            # Mock the ticket service
            query_processor_with_approval.ticket_service.get_or_create_ticket = (
                AsyncMock(return_value=sample_ticket)
            )

            # Test processing with approval requirement
            response = ""
            async for chunk in query_processor_with_approval.process(
                "user1", "Create a complex project"
            ):
                response += chunk

            # Verify simulation was run
            query_processor_with_approval.project_simulation_service.simulate_project.assert_called_once()

            # Verify ticket was submitted for approval
            query_processor_with_approval.project_approval_service.submit_for_approval.assert_called_once_with(
                sample_ticket
            )

            # Verify response contains simulation results and approval message
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


if __name__ == "__main__":
    pytest.main(["-xvs", "test_solana_agent.py"])
