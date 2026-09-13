import pytest

from finqa_eval.config import GeminiSettings
from finqa_eval.gemini_client import GenerationError, GeminiStructuredClient
from finqa_eval.schemas import DirectAnswerResponse


class FakeInteraction:
    def __init__(self, output_text: str | None) -> None:
        self.output_text = output_text


class FakeInteractions:
    def __init__(self, output_text: str | None) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeInteraction:
        self.calls.append(kwargs)
        return FakeInteraction(self.output_text)


class FakeClient:
    def __init__(self, output_text: str | None) -> None:
        self.interactions = FakeInteractions(output_text)


def make_client(output_text: str | None) -> tuple[GeminiStructuredClient, FakeClient]:
    fake_client = FakeClient(output_text)
    settings = GeminiSettings(api_key="test-key", model="test-model")

    return GeminiStructuredClient(settings, client=fake_client), fake_client


def test_generate_parses_valid_structured_response() -> None:
    client, fake_client = make_client('{"final_answer": "94"}')

    response = client.generate("What is 47 plus 47?", DirectAnswerResponse)

    assert response.final_answer == "94"
    assert fake_client.interactions.calls[0]["model"] == "test-model"
    assert fake_client.interactions.calls[0]["store"] is False


def test_generate_rejects_empty_output() -> None:
    client, _ = make_client(None)

    with pytest.raises(GenerationError, match="no text output"):
        client.generate("Question", DirectAnswerResponse)


def test_generate_rejects_invalid_schema_response() -> None:
    client, _ = make_client('{"wrong_field": "94"}')

    with pytest.raises(GenerationError, match="does not match"):
        client.generate("Question", DirectAnswerResponse)