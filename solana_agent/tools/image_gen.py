import logging
import base64
import io
import uuid
import asyncio
from typing import Dict, Any, List, Optional

# Solana Agent Kit imports
from solana_agent import AutoTool, ToolRegistry

# Required libraries
try:
    from openai import AsyncOpenAI
    import boto3  # type: ignore
    from botocore.exceptions import (  # type: ignore
        NoCredentialsError,
        PartialCredentialsError,
        ClientError,
    )

    BOTO3_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    boto3 = None  # type: ignore
    NoCredentialsError = None  # type: ignore
    PartialCredentialsError = None  # type: ignore
    ClientError = None  # type: ignore
    BOTO3_AVAILABLE = False
    logging.warning(
        "openai or boto3 library not found. ImageGenTool will not function. Install with 'pip install openai boto3'"
    )


# --- Setup Logger ---
logger = logging.getLogger(__name__)


class ImageGenTool(AutoTool):
    """
    Tool for generating images using OpenAI DALL-E and uploading them
    to S3-compatible storage.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize with auto-registration."""
        self._openai_model: str = "gpt-image-1"  # Default model define FIRST
        self._grok_model: str = "grok-2-image"  # Default model for Grok provider
        self._gemini_model: str = (
            "imagen-3.0-generate-002"  # Default model for Gemini provider
        )
        super().__init__(
            name="image_gen",
            description="Generates an image based on a text prompt using OpenAI DALL-E and uploads it to configured S3 storage, returning the public URL.",
            registry=registry,
        )
        self._openai_api_key: Optional[str] = None
        self._openai_base_url: Optional[str] = None  # Optional, depends on provider
        self._s3_endpoint_url: Optional[str] = None
        self._s3_access_key_id: Optional[str] = None
        self._s3_secret_access_key: Optional[str] = None
        self._s3_bucket_name: Optional[str] = None
        self._s3_region_name: Optional[str] = None  # Optional, depends on provider
        self._s3_public_url_base: Optional[str] = None  # Optional base for public URLs
        self._provider = "openai"  # Default provider

        logger.debug("ImageGenTool initialized.")

    def get_schema(self) -> Dict[str, Any]:
        """Define the tool schema."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed text description of the image to generate.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure OpenAI API key and S3 credentials."""
        super().configure(config)

        # 21 Configure S3 details from tool-specific config
        if (
            "tools" in config
            and isinstance(config["tools"], dict)
            and "image_gen" in config["tools"]
        ):
            tool_config = config["tools"]["image_gen"]
            if isinstance(tool_config, dict):
                self._openai_api_key = tool_config.get("api_key")
                if tool_config.get("provider") == "grok":
                    self._openai_base_url = "https://api.x.ai/v1"
                    self._provider = "grok"
                if tool_config.get("provider") == "gemini":
                    self._openai_base_url = (
                        "https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                    self._provider = "gemini"
                self._s3_endpoint_url = tool_config.get("s3_endpoint_url")
                self._s3_access_key_id = tool_config.get("s3_access_key_id")
                self._s3_secret_access_key = tool_config.get("s3_secret_access_key")
                self._s3_bucket_name = tool_config.get("s3_bucket_name")
                self._s3_region_name = tool_config.get("s3_region_name")  # Optional
                self._s3_public_url_base = tool_config.get(
                    "s3_public_url_base"
                )  # Optional
                self._openai_model = tool_config.get(
                    "default_model", self._openai_model
                )  # Allow overriding default model

                # Log configuration status
                if all(
                    [
                        self._s3_endpoint_url,
                        self._s3_access_key_id,
                        self._s3_secret_access_key,
                        self._s3_bucket_name,
                    ]
                ):
                    logger.info(
                        f"ImageGenTool: S3 configured for endpoint {self._s3_endpoint_url}, bucket {self._s3_bucket_name}."
                    )
                    if self._s3_public_url_base:
                        logger.info(
                            f"ImageGenTool: Using public URL base: {self._s3_public_url_base}"
                        )
                else:
                    logger.warning(
                        "ImageGenTool: S3 configuration is incomplete in config['tools']['image_gen']. Upload will fail."
                    )
            else:
                logger.warning(
                    "ImageGenTool: config['tools']['image_gen'] is not a dictionary."
                )

    def _is_configured(self) -> bool:
        """Check if all necessary configurations are set."""
        return bool(
            self._openai_api_key
            and self._s3_endpoint_url
            and self._s3_access_key_id
            and self._s3_secret_access_key
            and self._s3_bucket_name
        )

    def _upload_to_s3(
        self, image_bytes: bytes, object_key: str
    ) -> Optional[str]:  # pragma: no cover
        """Synchronous function to upload bytes to S3."""
        if not boto3:
            logger.error("Boto3 library not available for S3 upload.")
            return None

        try:
            session = boto3.session.Session()
            s3_client = session.client(
                service_name="s3",
                aws_access_key_id=self._s3_access_key_id,
                aws_secret_access_key=self._s3_secret_access_key,
                endpoint_url=self._s3_endpoint_url,
                region_name=self._s3_region_name,  # Pass region if configured
            )

            # Upload the image bytes
            s3_client.upload_fileobj(
                io.BytesIO(image_bytes),
                self._s3_bucket_name,
                object_key,
                ExtraArgs={
                    "ContentType": "image/png",
                    "ACL": "public-read",  # Make image publicly accessible
                },
            )
            logger.info(
                f"Successfully uploaded image to S3: s3://{self._s3_bucket_name}/{object_key}"
            )

            # Construct the public URL
            if self._s3_public_url_base:
                # Use custom base URL if provided (ensure it ends with /)
                base = (
                    self._s3_public_url_base
                    if self._s3_public_url_base.endswith("/")
                    else self._s3_public_url_base + "/"
                )
                public_url = base + object_key
            else:
                # Construct standard S3 URL based on the endpoint_url.
                # This format (https://<bucket>.<endpoint_domain>/<key>) works for many
                # S3-compatible providers like DigitalOcean Spaces.
                # It might need adjustment for providers with different URL structures (e.g., AWS S3).
                # Consider using s3_public_url_base config for guaranteed correctness.
                # region_part = f".{self._s3_region_name}" if self._s3_region_name else "" # Removed as it was unused in this logic
                endpoint_base = self._s3_endpoint_url.replace("https://", "")
                public_url = (
                    f"https://{self._s3_bucket_name}.{endpoint_base}/{object_key}"
                )

            logger.info(f"Public image URL: {public_url}")
            return public_url

        except (NoCredentialsError, PartialCredentialsError):
            logger.error("S3 credentials not found or incomplete.")
            return None
        except ClientError as e:
            logger.error(f"S3 ClientError during upload: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error during S3 upload: {e}")
            return None

    async def execute(  # pragma: no cover
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Generate image and upload to S3."""
        if not BOTO3_AVAILABLE or not AsyncOpenAI:
            return {
                "status": "error",
                "message": "Required libraries (openai, boto3) not available.",
            }

        if not self._is_configured():
            logger.error(
                "ImageGenTool is not fully configured (OpenAI key or S3 details missing)."
            )
            return {"status": "error", "message": "ImageGenTool not configured."}

        try:
            if self._provider == "grok":
                client = AsyncOpenAI(
                    api_key=self._openai_api_key,
                    base_url=self._openai_base_url,
                )
                effective_model = self._grok_model
                response = await client.images.generate(
                    model=effective_model,
                    prompt=prompt,
                    n=1,  # Generate one image
                    response_format="b64_json",
                )
                extension = "jpg"
            elif self._provider == "gemini":
                client = AsyncOpenAI(
                    api_key=self._openai_api_key,
                    base_url=self._openai_base_url,
                )
                effective_model = self._gemini_model
                response = await client.images.generate(
                    model=effective_model,
                    prompt=prompt,
                    n=1,  # Generate one image
                    response_format="b64_json",
                )
                extension = "png"
            else:
                client = AsyncOpenAI(api_key=self._openai_api_key)
                effective_model = self._openai_model
                response = await client.images.generate(
                    model=effective_model,
                    prompt=prompt,
                    n=1,  # Generate one image
                    size="1024x1024",  # Common size, make configurable if needed
                    output_format="jpeg",
                )
                extension = "jpeg"
            logger.info(
                f"Generating image with prompt: '{prompt[:50]}...' using model {effective_model}"
            )

            # 1. Generate image using OpenAI

            image_base64 = response.data[0].b64_json
            if not image_base64:
                logger.error("OpenAI did not return image data (b64_json).")
                return {
                    "status": "error",
                    "message": "Failed to get image data from OpenAI.",
                }

            image_bytes = base64.b64decode(image_base64)
            logger.debug(f"Image generated successfully ({len(image_bytes)} bytes).")

            # 2. Prepare filename and upload to S3 (in thread)
            unique_id = uuid.uuid4()
            safe_prefix = "image"
            object_key = f"{safe_prefix}_{unique_id}.{extension}"

            logger.debug(f"Uploading image to S3 object key: {object_key}")

            # Run synchronous boto3 upload in a separate thread
            public_url = await asyncio.to_thread(
                self._upload_to_s3, image_bytes, object_key
            )

            if public_url:
                return {
                    "status": "success",
                    "result": f"The image URL is {public_url}",
                }
            else:
                return {"status": "error", "message": "Failed to upload image to S3."}

        except Exception as e:
            logger.exception(f"Error during image generation or upload: {e}")
            return {
                "status": "error",
                "message": f"Image generation/upload failed: {type(e).__name__}: {e}",
            }


# --- Plugin Class ---


class ImageGenPlugin:
    """Plugin for integrating image generation and S3 upload."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "image_gen"
        self.config = None
        self.tool_registry = None
        self._tool: Optional[ImageGenTool] = None
        logger.info(f"Created ImageGenPlugin object with name: {self.name}")

    @property
    def description(self):
        """Return the plugin description."""
        return "Plugin providing image generation via OpenAI and upload to S3."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        """Initialize the plugin and register the tool."""
        if not BOTO3_AVAILABLE or not AsyncOpenAI:
            logger.warning(
                "ImageGenPlugin: Skipping initialization as openai or boto3 library is not available."
            )
            return

        self.tool_registry = tool_registry
        logger.info(f"Initializing {self.name} plugin.")

        # Create and register the tool
        self._tool = ImageGenTool(registry=tool_registry)

        # Verification (only if tool_registry is provided)
        if tool_registry is not None:
            registered_tool = tool_registry.get_tool(self.name)
            if registered_tool and isinstance(registered_tool, ImageGenTool):
                logger.info(f"{self.name} tool registration verification: Success")
            else:
                logger.error(
                    f"{self.name} tool registration verification: Failed or wrong type ({type(registered_tool)})"
                )

            all_tools = tool_registry.list_all_tools()
            logger.info(f"All registered tools after {self.name} init: {all_tools}")

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        """Configure the plugin and its underlying tool."""
        if not BOTO3_AVAILABLE or not AsyncOpenAI:
            logger.warning(
                "ImageGenPlugin: Skipping configuration as openai or boto3 library is not available."
            )
            return

        self.config = config
        logger.info(f"Configuring {self.name} plugin")

        if self._tool:
            self._tool.configure(self.config)
            logger.info(f"{self.name} tool configured.")
        else:
            logger.warning(
                f"Warning: {self.name} tool instance not found during configuration."
            )

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        """Return the list of tools provided by this plugin."""
        if not BOTO3_AVAILABLE or not AsyncOpenAI:
            return []
        return [self._tool] if self._tool else []


# --- Entry Point Function ---


def get_plugin():  # pragma: no cover
    """Return plugin instance for registration."""
    if not BOTO3_AVAILABLE or not AsyncOpenAI:
        logger.warning(
            "ImageGenPlugin: Cannot create plugin instance, openai or boto3 library not available."
        )

        # Return a dummy object
        class DummyPlugin:
            name = "image_gen (disabled)"
            description = "ImageGen plugin disabled (openai/boto3 library not found)"

            def initialize(self, *args, **kwargs):
                pass

            def configure(self, *args, **kwargs):
                pass

            def get_tools(self):
                return []

        return DummyPlugin()

    return ImageGenPlugin()
