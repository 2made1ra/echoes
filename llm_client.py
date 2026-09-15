"""Тонкая обёртка над OpenAI-совместимым API стороннего провайдера.

Один клиент, один ключ, один base_url: chat-модель и модель эмбеддингов
берутся из конфига. Ошибки провайдера наружу пробрасываются как есть —
маскировать их ответом от лица персонажа нельзя, иначе замеры дрейфа
перестанут отличать сетевой сбой от поломки характера.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from openai import AsyncOpenAI

from config import Settings, get_settings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "не задан LLM_API_KEY — скопируйте .env.example в .env и впишите ключ провайдера"
            )
        self._client = AsyncOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
        )

    async def chat(
        self,
        messages: Iterable[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        payload: list[Any] = list(messages)
        response = await self._client.chat.completions.create(
            model=model or self.settings.chat_model,
            messages=payload,
            temperature=self.settings.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.settings.max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("провайдер вернул пустой ответ")
        return content.strip()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def close(self) -> None:
        await self._client.close()
