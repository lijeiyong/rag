from dataclasses import dataclass
from typing import Any

from pgvector import Vector  # type: ignore[import-untyped]

from rag_app.database import Database


@dataclass(frozen=True, slots=True)
class SearchResult:
    source: str
    chunk_index: int
    content: str
    similarity: float


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def replace_source(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("each chunk must have exactly one embedding")

        async with self.database.connection() as connection, connection.transaction():
            # Replacing all chunks in one transaction avoids mixing old and new
            # versions of a document if ingestion fails halfway through.
            await connection.execute("DELETE FROM document_chunks WHERE source = %s", (source,))
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO document_chunks (source, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (source, index, chunk, Vector(embedding))
                        for index, (chunk, embedding) in enumerate(
                            zip(chunks, embeddings, strict=True)
                        )
                    ],
                )

    async def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """
                WITH query AS (SELECT %s::vector AS embedding)
                SELECT
                    source,
                    chunk_index,
                    content,
                    1 - (document_chunks.embedding <=> query.embedding) AS similarity
                FROM document_chunks, query
                ORDER BY document_chunks.embedding <=> query.embedding
                LIMIT %s
                """,
                (Vector(embedding), limit),
            )
            rows: list[dict[str, Any]] = await cursor.fetchall()

        # <=> is cosine distance. Converting it to 1 - distance makes larger
        # values mean "more similar," which is easier for API consumers to read.
        return [
            SearchResult(
                source=str(row["source"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]
