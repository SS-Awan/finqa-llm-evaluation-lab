import pytest

from finqa_eval.config import (
    DEFAULT_GEMINI_MODEL,
    ConfigurationError,
    load_gemini_settings,
)


def test_load_gemini_settings_reads_key_and_model() -> None:
    settings = load_gemini_settings(
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
        }
    )

    assert settings.api_key == "test-key"
    assert settings.model == "test-model"


def test_load_gemini_settings_uses_default_model() -> None:
    settings = load_gemini_settings({"GEMINI_API_KEY": "test-key"})

    assert settings.model == DEFAULT_GEMINI_MODEL


def test_load_gemini_settings_rejects_missing_key() -> None:
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
        load_gemini_settings({})


def test_load_gemini_settings_rejects_empty_model() -> None:
    with pytest.raises(ConfigurationError, match="GEMINI_MODEL cannot be empty"):
        load_gemini_settings(
            {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "   ",
            }
        )