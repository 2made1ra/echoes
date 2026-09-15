"""Long-term память: записи сессий в Qdrant.

Защита от мусорного ретрива строится на суммаризации (в индекс попадают
только записи, не сырые логи) плюс порог схожести. Ре-ранкинга в скоупе нет —
если ретрив всё равно мусорный, это результат эксперимента.

Память изолирована по пользователю и по персоне: одна персона не знает того,
что пользователь рассказывал другой, иначе ломаются границы знаний персонажа.
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient, models

from memory.record import Record

log = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, client: AsyncQdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    @classmethod
    def from_url(cls, url: str, collection: str, dim: int) -> "LongTermMemory":
        return cls(AsyncQdrantClient(url=url), collection, dim)

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
        )
        for field in ("user_id", "persona"):
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        log.info("создана коллекция %s (dim=%s)", self._collection, self._dim)

    async def store(self, record: Record, vector: list[float]) -> None:
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=record.record_id,
                    vector=vector,
                    payload=record.model_dump(mode="json"),
                )
            ],
        )

    async def search(
        self,
        user_id: str,
        persona: str,
        vector: list[float],
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[Record]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="persona", match=models.MatchValue(value=persona)),
                ]
            ),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [Record.model_validate(point.payload) for point in response.points if point.payload]

    async def close(self) -> None:
        await self._client.close()
