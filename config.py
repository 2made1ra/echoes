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
    # Бюджет токенов на реплику. У reasoning-моделей в него входят и
    # рассуждения (замер: до ~190 из ~280 на реплику), а при нехватке провайдер
    # отдаёт пустой ответ. Длину реплики держат карточка и переинжект, не лимит.
    max_tokens: int = 1000

    # Суммаризация сессии в запись памяти: служебный вызов, нужна точность,
    # а не разнообразие. Бюджет с запасом на рассуждения: на сессии из 16
    # реплик они доходили до ~440 токенов при ~600 всего и растут с длиной
    # лога; при нехватке провайдер отдаёт пустой ответ и закрытие падает с 502.
    summarizer_temperature: float = 0.0
    summarizer_max_tokens: int = 2000

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "memories"

    # Сколько дней живут история сессии и колода первых реплик в Redis.
    session_ttl_days: int = 30
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

    # Ограничения входа API: длины идентификаторов и реплики пользователя.
    api_user_id_max_chars: int = 128
    api_session_id_max_chars: int = 128
    api_persona_max_chars: int = 64
    api_message_max_chars: int = 4000

    # Меньше стольких сообщений в истории — сессию нечего сжимать в запись.
    archive_min_messages: int = 2
    # Длина отпечатка набора первых реплик в ключе колоды Redis.
    deck_fingerprint_chars: int = 16
    # Сколько символов сырого ответа модели показывать в логах и отчётах.
    log_preview_chars: int = 200

    # Замер дрейфа: доля сохранённых маркеров считается по окнам из
    # EVAL_WINDOW реплик, всего EVAL_WINDOWS окон (8 × 3 → 1–8 / 9–16 / 17–24).
    eval_window: int = 8
    eval_windows: int = 3
    # Диалогов на условие: один диалог слишком шумный, доля по окнам
    # усредняется по нескольким. Больше — точнее, но дороже.
    eval_runs: int = 3
    # Сколько диалогов замера идёт к провайдеру одновременно.
    eval_concurrency: int = 4
    # Сид мок-модели (--mock): прогон на моке воспроизводим.
    eval_seed: int = 0
    # Мок-модель (--mock): шанс сломать роль на первой реплике после
    # переинжекта, прирост шанса за каждую следующую и доля сбоев провайдера.
    eval_mock_break_base: float = 0.05
    eval_mock_break_per_turn: float = 0.04
    eval_mock_error_rate: float = 0.0

    # Каждая поддиректория persona/ (кроме _shared) — отдельная персона.
    persona_dir: Path = ROOT / "persona"
    # Персона новой сессии, если клиент не указал её явно.
    # Не задана — первая персона из persona/ по алфавиту.
    default_persona: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
