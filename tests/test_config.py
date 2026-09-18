from pathlib import Path

from pytest import MonkeyPatch

from rag_app.config import Settings
from rag_app.providers import GeminiProvider, OpenAIProvider, create_provider


def test_settings_load_values_from_dotenv(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Process environment wins over .env, so remove Compose-injected values to
    # demonstrate the .env layer itself.
    for variable in ("AI_PROVIDER", "OPENAI_API_KEY", "OPENAI_ANSWER_MODEL", "CHUNK_SIZE"):
        monkeypatch.delenv(variable, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_PROVIDER=openai\n"
        "OPENAI_API_KEY=from-dotenv\n"
        "OPENAI_ANSWER_MODEL=test-answer-model\n"
        "CHUNK_SIZE=321\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.ai_provider == "openai"
    assert settings.openai_api_key == "from-dotenv"
    assert settings.openai_answer_model == "test-answer-model"
    assert settings.chunk_size == 321


def test_gemini_is_the_default_provider() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert isinstance(create_provider(settings), GeminiProvider)


def test_openai_provider_remains_selectable() -> None:
    settings = Settings(ai_provider="openai", _env_file=None)  # type: ignore[call-arg]

    assert isinstance(create_provider(settings), OpenAIProvider)
