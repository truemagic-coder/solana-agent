import logging
from typing import Dict, Any, Optional, List
import scrubadub
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
)

logger = logging.getLogger(__name__)


class PII(InputGuardrail, OutputGuardrail):
    """
    A guardrail using Scrubadub to detect and remove PII.

    Requires 'scrubadub'. Install with: pip install solana-agent[guardrails]
    """

    DEFAULT_REPLACEMENT = "[REDACTED_{detector_name}]"
    DEFAULT_LANG = "en_US"  # Scrubadub uses locale format

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.replacement_format = self.config.get(
            "replacement", self.DEFAULT_REPLACEMENT
        )
        self.locale = self.config.get("locale", self.DEFAULT_LANG)
        # Optional: Specify detectors to use, None uses defaults
        self.detector_list: Optional[List[str]] = self.config.get("detectors")
        # Optional: Add custom detectors if needed via config
        self.extra_detector_list = self.config.get(
            "extra_detectors", []
        )  # List of detector classes/instances

        try:
            # Initialize Scrubber
            # Note: detector_list expects instances, not names. Need mapping or direct instantiation if customizing.
            # For simplicity, we'll use defaults or allow passing instances via config (advanced).
            # Using default detectors if self.detector_list is None.
            if self.detector_list is not None:
                logger.warning(
                    "Customizing 'detectors' by name list is not directly supported here yet. Using defaults."
                )
                # TODO: Add logic to map names to detector classes if needed.
                self.scrubber = scrubadub.Scrubber(locale=self.locale)
            else:
                self.scrubber = scrubadub.Scrubber(locale=self.locale)

            # Add any extra detectors passed via config (e.g., custom regex detectors)
            for detector in self.extra_detector_list:
                # Assuming extra_detectors are already instantiated objects
                # Or add logic here to instantiate them based on class paths/names
                if isinstance(detector, scrubadub.detectors.Detector):
                    self.scrubber.add_detector(detector)
                else:
                    logger.warning(f"Invalid item in extra_detectors: {detector}")

            logger.info(f"ScrubadubPIIFilter initialized for locale '{self.locale}'")

        except ImportError:
            logger.error(
                "Scrubadub not installed. Please install with 'pip install solana-agent[guardrails]'"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Scrubadub: {e}", exc_info=True)
            raise

    async def process(self, text: str) -> str:
        """Clean text using Scrubadub."""
        try:
            # Scrubadub's clean method handles the replacement logic.
            # We need to customize the replacement format per detector.
            # This requires iterating through filth found first.

            clean_text = text
            filth_list = list(self.scrubber.iter_filth(text))  # Get all findings

            if not filth_list:
                return text

            # Sort by start index to handle replacements correctly
            filth_list.sort(key=lambda f: f.beg)

            offset = 0
            for filth in filth_list:
                start = filth.beg + offset
                end = filth.end + offset
                replacement_text = self.replacement_format.format(
                    detector_name=filth.detector_name,
                    text=filth.text,
                    locale=filth.locale,
                    # Add other filth attributes if needed in format string
                )

                clean_text = clean_text[:start] + replacement_text + clean_text[end:]
                offset += len(replacement_text) - (filth.end - filth.beg)

            if clean_text != text:
                logger.debug(
                    f"ScrubadubPIIFilter redacted {len(filth_list)} pieces of filth."
                )
            return clean_text

        except Exception as e:
            logger.error(f"Error during Scrubadub cleaning: {e}", exc_info=True)
            return text  # Return original text on error
