from collections.abc import Sequence
from typing import Protocol

from rag_app.chunking import chunk_text
from rag_app.repository import SearchResult


class AIProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def answer(self, question: str, contexts: Sequence[str]) -> str: ...


class ChunkRepository(Protocol):
    async def replace_source(
        self, source: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None: ...

    async def search(self, embedding: list[float], limit: int) -> list[SearchResult]: ...


class NoDocumentsError(RuntimeError):
    pass


class RagService:
    def __init__(
        self,
        provider: AIProvider,
        repository: ChunkRepository,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest(self, source: str, content: str) -> int:
        chunks = chunk_text(content, size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            raise ValueError("document content must contain non-whitespace text")

        # Embedding chunks in one batch reduces API round trips and produces the
        # semantic coordinates that pgvector will compare during retrieval.
        embeddings = await self.provider.embed(chunks)
        await self.repository.replace_source(source, chunks, embeddings)
        return len(chunks)

    async def ask(self, question: str, top_k: int) -> tuple[str, list[SearchResult]]:
        # Questions and documents must use the same embedding model/dimensions;
        # otherwise their vectors do not share a meaningful coordinate space.
        query_embedding = (await self.provider.embed([question]))[0]
        matches = await self.repository.search(query_embedding, top_k)
        if not matches:
            raise NoDocumentsError("No documents have been indexed yet.")

        answer = await self.provider.answer(question, [match.content for match in matches])
        return answer, matches
