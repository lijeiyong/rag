from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # These field values are fallbacks, not forced configuration. BaseSettings
    # replaces them with process environment variables or values from `.env`.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://rag:rag@db:5432/rag"
    ai_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_answer_model: str = "gemini-3.8-flash"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_answer_model: str = "gpt-5-mini"
    embedding_dimensions: int = Field(default=1536, ge=1)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
