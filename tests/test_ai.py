import datetime
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch


from solana_agent.ai import (
    # Domain Models
    TicketStatus, Ticket, MemoryInsight, TicketResolution,

    # Service classes
    AgentService, RoutingService, TicketService, HandoffService, NPSService,
    MemoryService, CriticService, QueryProcessor,

    # Repository implementations
    MongoTicketRepository, MongoHandoffRepository, MongoNPSSurveyRepository,
    MongoMemoryRepository,

    # Adapters
    MongoDBAdapter, OpenAIAdapter, ZepMemoryAdapter, PineconeAdapter,

    # Factory
    SolanaAgentFactory,

    # Client interface
    SolanaAgent
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
    repo.get_metrics = MagicMock(return_value={
        "nps_score": 75,
        "promoters": 8,
        "passives": 2,
        "detractors": 0,
        "total_responses": 10,
        "avg_score": 8.5
    })

    return repo


@pytest.fixture
def mock_memory_repository(mock_mongodb_adapter, mock_vector_store):
    """Create a mocked memory repository."""
    repo = MongoMemoryRepository(mock_mongodb_adapter, mock_vector_store)

    # Override methods for testing
    repo.store_insight = MagicMock(return_value=str(uuid.uuid4()))
    repo.search = MagicMock(return_value=[
        {
            "id": "123",
            "fact": "Test fact",
            "relevance": "Test relevance",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    ])

    return repo


@pytest.fixture
def agent_service(mock_llm_provider):
    """Create an agent service for testing."""
    service = AgentService(mock_llm_provider)

    # Register test agents
    service.register_ai_agent(
        "test_agent",
        "You are a test agent.",
        "General testing",
        "gpt-4o-mini"
    )

    service.register_ai_agent(
        "solana_specialist",
        "You are a Solana blockchain specialist.",
        "Solana blockchain",
        "gpt-4o"
    )

    service.register_human_agent(
        "human1",
        "Human Agent",
        "Complex issues",
        None  # No notification handler for testing
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
    return HandoffService(mock_handoff_repository, mock_ticket_repository, agent_service)


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
def query_processor(
    agent_service,
    routing_service,
    ticket_service,
    handoff_service,
    memory_service,
    nps_service,
    critic_service,
    mock_memory_provider
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
        router_model="gpt-4o-mini"
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


#############################################
# TESTS
#############################################

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
        async for chunk in agent_service.generate_response("nonexistent_agent", "user1", "Hi"):
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
        agent_name = await routing_service.route_query("Tell me about Solana blockchain")

        # Since our mock doesn't actually make decisions, we just check the return type
        assert isinstance(agent_name, str)

    def test_match_agent_name(self, routing_service):
        """Test matching agent names from responses."""
        agent_names = ["test_agent", "solana_specialist"]

        # Exact match
        assert routing_service._match_agent_name(
            "test_agent", agent_names) == "test_agent"

        # Case insensitive match
        assert routing_service._match_agent_name(
            "TEST_AGENT", agent_names) == "test_agent"

        # Partial match
        assert routing_service._match_agent_name(
            "something with solana_specialist in it", agent_names) == "solana_specialist"

        # No match should default to first agent
        assert routing_service._match_agent_name(
            "no match", agent_names) == "test_agent"


class TestTicketService:
    """Tests for the TicketService."""

    @pytest.mark.asyncio
    async def test_get_or_create_ticket_new(self, ticket_service, mock_ticket_repository):
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
    async def test_get_or_create_ticket_existing(self, ticket_service, mock_ticket_repository, sample_ticket):
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
            "123", TicketStatus.ACTIVE, assigned_to="test_agent")

        # Verify the repository was called with correct parameters
        mock_ticket_repository.update.assert_called_once()
        args, kwargs = mock_ticket_repository.update.call_args
        assert args[0] == "123"
        assert args[1]["status"] == TicketStatus.ACTIVE
        assert args[1]["assigned_to"] == "test_agent"
        assert "updated_at" in args[1]

    def test_mark_ticket_resolved(self, ticket_service, mock_ticket_repository):
        """Test marking a ticket as resolved."""
        resolution_data = {
            "confidence": 0.9,
            "reasoning": "Issue was fully addressed"
        }

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
    async def test_process_handoff(self, handoff_service, mock_handoff_repository, mock_ticket_repository, sample_ticket):
        """Test processing a handoff between agents."""
        # Configure mocks
        mock_ticket_repository.get_by_id.return_value = sample_ticket

        # Test handoff process
        result = await handoff_service.process_handoff("123", "test_agent", "solana_specialist", "Needs blockchain expertise")

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
    async def test_process_handoff_ticket_not_found(self, handoff_service, mock_ticket_repository):
        """Test handling a handoff when the ticket doesn't exist."""
        # Configure mock to return no ticket
        mock_ticket_repository.get_by_id.return_value = None

        # Test that ValueError is raised
        with pytest.raises(ValueError, match="Ticket .* not found"):
            await handoff_service.process_handoff("123", "test_agent", "solana_specialist", "Needs blockchain expertise")

    @pytest.mark.asyncio
    async def test_process_handoff_invalid_agent(self, handoff_service, mock_ticket_repository, sample_ticket):
        """Test handling a handoff when the target agent doesn't exist."""
        # Configure mock to return a ticket
        mock_ticket_repository.get_by_id.return_value = sample_ticket

        # Test that ValueError is raised
        with pytest.raises(ValueError, match="Target agent .* not found"):
            await handoff_service.process_handoff("123", "test_agent", "nonexistent_agent", "Needs expertise")


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
            "survey123", 9, "Great service!")

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
        mock_response = '{"insights": [{"fact": "Test fact", "relevance": "Test relevance"}]}'

        # We need to patch the generate_text method to return our mock JSON
        async def mock_generate_text(*args, **kwargs):
            yield mock_response

        mock_llm_provider.generate_text = mock_generate_text

        # Test insight extraction
        insights = await memory_service.extract_insights("user1", {"message": "Hi", "response": "Hello"})

        assert len(insights) == 1
        assert insights[0].fact == "Test fact"
        assert insights[0].relevance == "Test relevance"

    @pytest.mark.asyncio
    async def test_store_insights(self, memory_service, mock_memory_repository):
        """Test storing insights in memory."""
        insights = [
            MemoryInsight(fact="Test fact 1", relevance="Test relevance 1"),
            MemoryInsight(fact="Test fact 2", relevance="Test relevance 2")
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
            return_value="System command result")

        # Test processing a system command
        response = ""
        async for chunk in query_processor.process("user1", "!command"):
            response += chunk

        assert response == "System command result"
        query_processor._process_system_commands.assert_called_once()

    async def test_process_new_ticket(self, query_processor, mock_ticket_repository, agent_service):
        """Test processing a message creating a new ticket."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        query_processor.routing_service.route_query = AsyncMock(
            return_value="test_agent")
        query_processor._assess_task_complexity = AsyncMock(
            return_value={"t_shirt_size": "M"})
        mock_ticket_repository.get_active_for_user.return_value = None

        # Mock the ticket service methods directly
        query_processor.ticket_service.update_ticket_status = MagicMock(
            return_value=True)

        # Mock the ticket
        new_ticket = Ticket(
            id="new_ticket_id",
            user_id="user1",
            query="Help me",
            status=TicketStatus.NEW,
            assigned_to="",
            created_at=datetime.datetime.now(datetime.timezone.utc)
        )
        query_processor.ticket_service.get_or_create_ticket = AsyncMock(
            return_value=new_ticket)

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
        # Changed to assert syntax
        assert query_processor.ticket_service.update_ticket_status.called

    async def test_process_existing_ticket(self, query_processor, mock_ticket_repository, sample_ticket):
        """Test processing a message for an existing ticket."""
        # Setup required mocks
        query_processor._is_human_agent = AsyncMock(return_value=False)
        query_processor._is_simple_greeting = AsyncMock(return_value=False)
        query_processor._process_system_commands = AsyncMock(return_value=None)
        query_processor.routing_service.route_query = AsyncMock(
            return_value="test_agent")
        mock_ticket_repository.get_active_for_user.return_value = sample_ticket

        # Mock service methods
        query_processor.ticket_service.update_ticket_status = MagicMock(
            return_value=True)
        query_processor.nps_service.create_survey = MagicMock(
            return_value="survey123")

        # Mock ticket resolution check
        resolution = TicketResolution(
            status="resolved",
            confidence=0.8,
            reasoning="Issue was resolved"
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
        assert query_processor.nps_service.create_survey.called  # Changed to assert syntax

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


class TestPlatformIntegration:
    """Integration tests for the entire platform."""

    @pytest.mark.asyncio
    async def test_agent_factory_creation(self):
        """Test creating the agent system from configuration."""
        # Create a minimal configuration
        config = {
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db"
            },
            "openai": {
                "api_key": "test_key",
                "default_model": "gpt-4o-mini"
            },
            "enable_critic": True,
            "router_model": "gpt-4o-mini",
            "agents": [
                {
                    "name": "general",
                    "instructions": "You are a helpful assistant.",
                    "specialization": "General assistance"
                }
            ]
        }

        # Mock the adapters
        with patch("solana_agent.ai.MongoDBAdapter") as mock_mongo, \
                patch("solana_agent.ai.OpenAIAdapter") as mock_openai:

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
                "database": "test_db"
            },
            "openai": {
                "api_key": "test_key",
                "default_model": "gpt-4o-mini"
            },
            "agents": []
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


if __name__ == "__main__":
    pytest.main(["-xvs", "test_solana_agent.py"])
