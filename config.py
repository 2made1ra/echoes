"""Конфигурация сервиса. Все секреты — только из окружения/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Один ключ и один base_url на всё: и chat, и embeddings.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    temperature: float = 0.9
    max_tokens: int = 400

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "memories"

    # Сколько последних сообщений сессии уходит в промпт.
    session_window: int = 20
    # N переинжекта: блок напоминания черт каждые N реплик пользователя.
    # Стартовое значение 6, подбирается замером из eval/. Это значение по
    # умолчанию: персона может переопределить его в своём persona.toml.
    reinject_every: int = 6
    # Сколько записей подтягивается из long-term памяти.
    retrieval_top_k: int = 3
    # Ниже этого порога схожести запись считается мусорной и не подтягивается.
    retrieval_score_threshold: float = 0.3

    # Каждая поддиректория persona/ (кроме _shared) — отдельная персона.
    persona_dir: Path = ROOT / "persona"
    # Персона новой сессии, если клиент не указал её явно.
    # Не задана — первая персона из persona/ по алфавиту.
    default_persona: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
