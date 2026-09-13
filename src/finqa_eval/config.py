"""Local configuration for Gemini access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class ConfigurationError(ValueError):
    """Raised when required local configuration is missing."""


@dataclass(frozen=True)
class GeminiSettings:
    """The Gemini settings required by the evaluation runner."""

    api_key: str
    model: str


def load_gemini_settings(
    environment: Mapping[str, str] | None = None,
) -> GeminiSettings:
    """Load Gemini settings without ever printing the API key."""
    if environment is None:
        load_dotenv()
        environment = os.environ

    api_key = environment.get("GEMINI_API_KEY", "").strip()
    model = environment.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()

    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is required in the local .env file.")

    if not model:
        raise ConfigurationError("GEMINI_MODEL cannot be empty.")

    return GeminiSettings(api_key=api_key, model=model)