"""
Tests for Image Generation Tool.

Tests the ImageGenTool which generates images using OpenAI/Grok/Gemini
and uploads them to S3-compatible storage.
"""

import base64
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from solana_agent.tools.image_gen import (
    ImageGenTool,
    ImageGenPlugin,
    get_plugin,
    BOTO3_AVAILABLE,
)


# Test image data (1x1 PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def configured_tool():
    """Create a fully configured ImageGenTool."""
    tool = ImageGenTool()
    tool.configure(
        {
            "tools": {
                "image_gen": {
                    "api_key": "test-key",
                    "s3_endpoint_url": "https://s3.example.com",
                    "s3_access_key_id": "access-key",
                    "s3_secret_access_key": "secret-key",
                    "s3_bucket_name": "test-bucket",
                }
            }
        }
    )
    return tool


class TestImageGenToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self):
        """Should have correct tool name."""
        tool = ImageGenTool()
        assert tool.name == "image_gen"

    def test_schema_has_required_properties(self):
        """Should include prompt in required properties."""
        tool = ImageGenTool()
        schema = tool.get_schema()
        assert "prompt" in schema["properties"]
        assert "prompt" in schema["required"]

    def test_schema_prompt_description(self):
        """Should have descriptive prompt field."""
        tool = ImageGenTool()
        schema = tool.get_schema()
        assert "description" in schema["properties"]["prompt"]

    def test_schema_no_additional_properties(self):
        """Should not allow additional properties."""
        tool = ImageGenTool()
        schema = tool.get_schema()
        assert schema.get("additionalProperties") is False

    def test_default_provider_is_openai(self):
        """Should default to openai provider."""
        tool = ImageGenTool()
        assert tool._provider == "openai"

    def test_default_models(self):
        """Should have default models for each provider."""
        tool = ImageGenTool()
        assert tool._openai_model == "gpt-image-1"
        assert tool._grok_model == "grok-2-image"
        assert tool._gemini_model == "imagen-3.0-generate-002"


class TestImageGenToolConfigure:
    """Test configuration method."""

    def test_configure_stores_api_key(self):
        """Should store API key from config."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-openai-key",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._openai_api_key == "test-openai-key"

    def test_configure_stores_s3_config(self):
        """Should store S3 configuration."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                        "s3_region_name": "us-east-1",
                    }
                }
            }
        )
        assert tool._s3_endpoint_url == "https://s3.example.com"
        assert tool._s3_bucket_name == "test-bucket"
        assert tool._s3_region_name == "us-east-1"

    def test_configure_grok_provider(self):
        """Should configure Grok provider correctly."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "provider": "grok",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._provider == "grok"
        assert tool._openai_base_url == "https://api.x.ai/v1"

    def test_configure_gemini_provider(self):
        """Should configure Gemini provider correctly."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "provider": "gemini",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._provider == "gemini"
        assert "generativelanguage.googleapis.com" in tool._openai_base_url

    def test_configure_custom_model(self):
        """Should allow custom model configuration."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "default_model": "custom-model-v2",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._openai_model == "custom-model-v2"

    def test_configure_public_url_base(self):
        """Should store public URL base for S3."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                        "s3_public_url_base": "https://cdn.example.com/images",
                    }
                }
            }
        )
        assert tool._s3_public_url_base == "https://cdn.example.com/images"

    def test_configure_empty_tools_config(self):
        """Should handle empty tools config gracefully."""
        tool = ImageGenTool()
        tool.configure({"tools": {}})
        assert tool._openai_api_key is None

    def test_configure_non_dict_tool_config(self):
        """Should handle non-dict tool config gracefully."""
        tool = ImageGenTool()
        tool.configure({"tools": {"image_gen": "invalid"}})
        assert tool._openai_api_key is None


class TestImageGenToolIsConfigured:
    """Test configuration validation."""

    def test_is_configured_returns_false_when_incomplete(self):
        """Should return False when config is incomplete."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        # Missing S3 config
                    }
                }
            }
        )
        assert tool._is_configured() is False

    def test_is_configured_returns_true_when_complete(self):
        """Should return True when config is complete."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._is_configured() is True

    def test_is_configured_false_without_api_key(self):
        """Should return False when API key is missing."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )
        assert tool._is_configured() is False


class TestImageGenToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_not_configured_error(self):
        """Should return error when not configured."""
        tool = ImageGenTool()
        tool.configure({"tools": {"image_gen": {}}})

        result = await tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "error"
        assert "not configured" in result["message"].lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_execute_success_openai_provider(self, configured_tool):
        """Should successfully generate and upload image with OpenAI."""
        # Mock OpenAI response
        mock_image_data = MagicMock()
        mock_image_data.b64_json = TEST_IMAGE_B64

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        # Mock S3 upload
        with (
            patch("solana_agent.tools.image_gen.AsyncOpenAI", return_value=mock_client),
            patch.object(
                configured_tool,
                "_upload_to_s3",
                return_value="https://cdn.example.com/image_test.jpeg",
            ),
        ):
            result = await configured_tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "success"
        assert "https://cdn.example.com/image_test.jpeg" in result["result"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_execute_no_image_data_error(self, configured_tool):
        """Should return error when OpenAI returns no image data."""
        # Mock OpenAI response with no image data
        mock_image_data = MagicMock()
        mock_image_data.b64_json = None

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch(
            "solana_agent.tools.image_gen.AsyncOpenAI", return_value=mock_client
        ):
            result = await configured_tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "error"
        assert "image data" in result["message"].lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_execute_s3_upload_failure(self, configured_tool):
        """Should return error when S3 upload fails."""
        # Mock OpenAI response
        mock_image_data = MagicMock()
        mock_image_data.b64_json = TEST_IMAGE_B64

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        # Mock S3 upload failure
        with (
            patch("solana_agent.tools.image_gen.AsyncOpenAI", return_value=mock_client),
            patch.object(configured_tool, "_upload_to_s3", return_value=None),
        ):
            result = await configured_tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "error"
        assert "s3" in result["message"].lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_execute_exception_handling(self, configured_tool):
        """Should handle exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )

        with patch(
            "solana_agent.tools.image_gen.AsyncOpenAI", return_value=mock_client
        ):
            result = await configured_tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "error"
        assert "API rate limit exceeded" in result["message"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_execute_grok_provider(self):
        """Should use Grok provider settings."""
        tool = ImageGenTool()
        tool.configure(
            {
                "tools": {
                    "image_gen": {
                        "api_key": "test-key",
                        "provider": "grok",
                        "s3_endpoint_url": "https://s3.example.com",
                        "s3_access_key_id": "access-key",
                        "s3_secret_access_key": "secret-key",
                        "s3_bucket_name": "test-bucket",
                    }
                }
            }
        )

        mock_image_data = MagicMock()
        mock_image_data.b64_json = TEST_IMAGE_B64

        mock_response = MagicMock()
        mock_response.data = [mock_image_data]

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with (
            patch(
                "solana_agent.tools.image_gen.AsyncOpenAI", return_value=mock_client
            ) as mock_openai,
            patch.object(
                tool,
                "_upload_to_s3",
                return_value="https://cdn.example.com/image.jpg",
            ),
        ):
            result = await tool.execute(prompt="A beautiful sunset")

        assert result["status"] == "success"
        # Verify Grok base URL was used
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.x.ai/v1",
        )


class TestImageGenToolUploadToS3:
    """Test S3 upload functionality."""

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_upload_to_s3_success(self, configured_tool):
        """Should successfully upload to S3."""
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client

        with patch(
            "solana_agent.tools.image_gen.boto3.session.Session",
            return_value=mock_session,
        ):
            image_bytes = base64.b64decode(TEST_IMAGE_B64)
            result = configured_tool._upload_to_s3(image_bytes, "test-image.png")

        assert result is not None
        assert "test-bucket" in result
        assert "test-image.png" in result

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_upload_to_s3_with_public_url_base(self):
        """Should use public URL base when configured."""
        tool = ImageGenTool()
        tool._s3_endpoint_url = "https://s3.example.com"
        tool._s3_access_key_id = "access-key"
        tool._s3_secret_access_key = "secret-key"
        tool._s3_bucket_name = "test-bucket"
        tool._s3_region_name = None
        tool._s3_public_url_base = "https://cdn.example.com"

        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client

        with patch(
            "solana_agent.tools.image_gen.boto3.session.Session",
            return_value=mock_session,
        ):
            image_bytes = base64.b64decode(TEST_IMAGE_B64)
            result = tool._upload_to_s3(image_bytes, "test-image.png")

        assert result == "https://cdn.example.com/test-image.png"

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_upload_to_s3_with_trailing_slash_url_base(self):
        """Should handle trailing slash in public URL base."""
        tool = ImageGenTool()
        tool._s3_endpoint_url = "https://s3.example.com"
        tool._s3_access_key_id = "access-key"
        tool._s3_secret_access_key = "secret-key"
        tool._s3_bucket_name = "test-bucket"
        tool._s3_region_name = None
        tool._s3_public_url_base = "https://cdn.example.com/"

        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client

        with patch(
            "solana_agent.tools.image_gen.boto3.session.Session",
            return_value=mock_session,
        ):
            image_bytes = base64.b64decode(TEST_IMAGE_B64)
            result = tool._upload_to_s3(image_bytes, "test-image.png")

        assert result == "https://cdn.example.com/test-image.png"

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_upload_to_s3_client_error(self, configured_tool):
        """Should return None on S3 ClientError."""
        from botocore.exceptions import ClientError

        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_s3_client.upload_fileobj.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutObject",
        )
        mock_session.client.return_value = mock_s3_client

        with patch(
            "solana_agent.tools.image_gen.boto3.session.Session",
            return_value=mock_session,
        ):
            image_bytes = base64.b64decode(TEST_IMAGE_B64)
            result = configured_tool._upload_to_s3(image_bytes, "test-image.png")

        assert result is None


class TestImageGenPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = ImageGenPlugin()
        assert plugin.name == "image_gen"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = ImageGenPlugin()
        assert "image" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = ImageGenPlugin()
        assert plugin.get_tools() == []

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = ImageGenPlugin()
        mock_registry = MagicMock()
        mock_registry.get_tool.return_value = None

        plugin.initialize(mock_registry)

        assert plugin._tool is not None

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_plugin_configure(self):
        """Should configure the tool after initialization."""
        plugin = ImageGenPlugin()
        mock_registry = MagicMock()
        mock_registry.get_tool.return_value = None
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "image_gen": {
                    "api_key": "test-key",
                    "s3_endpoint_url": "https://s3.example.com",
                    "s3_access_key_id": "access-key",
                    "s3_secret_access_key": "secret-key",
                    "s3_bucket_name": "test-bucket",
                }
            }
        }
        plugin.configure(config)

        assert plugin.config == config
        assert plugin._tool._openai_api_key == "test-key"

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_plugin_get_tools_after_init(self):
        """Should return tool list after initialization."""
        plugin = ImageGenPlugin()
        mock_registry = MagicMock()
        mock_registry.get_tool.return_value = None
        plugin.initialize(mock_registry)

        tools = plugin.get_tools()
        assert len(tools) == 1


class TestGetPlugin:
    """Test get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return an ImageGenPlugin instance or dummy."""
        plugin = get_plugin()
        assert plugin.name == "image_gen" or "disabled" in plugin.name

    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    def test_get_plugin_returns_real_plugin(self):
        """Should return an ImageGenPlugin instance when libs available."""
        plugin = get_plugin()
        assert isinstance(plugin, ImageGenPlugin)
        assert plugin.name == "image_gen"
