"""Gemini client wrapper for validated structured responses."""

from __future__ import annotations

from typing import Any, TypeVar

from google import genai
from pydantic import BaseModel, ValidationError

from finqa_eval.config import GeminiSettings


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class GenerationError(RuntimeError):
    """Raised when Gemini does not return a valid structured response."""


class GeminiStructuredClient:
    """Generate and validate one structured response at a time."""

    def __init__(self, settings: GeminiSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = (
            genai.Client(api_key=settings.api_key) if client is None else client
        )

    def generate(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Send one prompt and parse Gemini's JSON response into a Pydantic model."""
        interaction = self._client.interactions.create(
            model=self.settings.model,
            input=prompt,
            store=False,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_model.model_json_schema(),
            },
        )

        output_text = getattr(interaction, "output_text", None)

        if not isinstance(output_text, str) or not output_text.strip():
            raise GenerationError("Gemini returned no text output.")

        try:
            return response_model.model_validate_json(output_text)
        except ValidationError as error:
            raise GenerationError(
                "Gemini returned text that does not match the required response schema."
            ) from error