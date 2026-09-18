from collections.abc import Sequence

import pytest

from rag_app.rag import NoDocumentsError, RagService
from rag_app.repository import SearchResult


class FakeProvider:
    def __init__(self) -> None:
        self.embedded: list[list[str]] = []
        self.answer_contexts: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    async def answer(self, question: str, contexts: Sequence[str]) -> str:
        self.answer_contexts = list(contexts)
        return f"Answered: {question}"


class FakeRepository:
    def __init__(self, search_results: list[SearchResult] | None = None) -> None:
        self.search_results = search_results or []
        self.saved: tuple[str, list[str], list[list[float]]] | None = None

    async def replace_source(
        self, source: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        self.saved = (source, chunks, embeddings)

    async def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
        # This slice models SQL LIMIT: top_k caps how many closest chunks reach the LLM.
        return self.search_results[:limit]


@pytest.mark.asyncio
async def test_ingest_chunks_embeds_and_saves_document() -> None:
    provider = FakeProvider()
    repository = FakeRepository()
    service = RagService(provider, repository, chunk_size=100, chunk_overlap=20)

    count = await service.ingest("guide", " ".join(["retrieval"] * 30))

    assert count > 1
    assert repository.saved is not None
    source, chunks, embeddings = repository.saved
    assert source == "guide"
    assert len(chunks) == len(embeddings) == count


@pytest.mark.asyncio
async def test_ask_retrieves_context_before_generating_answer() -> None:
    provider = FakeProvider()
    match = SearchResult("guide", 0, "pgvector searches embeddings", 0.92)
    service = RagService(provider, FakeRepository([match]), chunk_size=100, chunk_overlap=20)

    answer, sources = await service.ask("What does pgvector do?", top_k=3)

    assert answer == "Answered: What does pgvector do?"
    assert sources == [match]
    assert provider.answer_contexts == [match.content]


@pytest.mark.asyncio
async def test_top_k_limits_context_sent_to_answer_model() -> None:
    provider = FakeProvider()
    matches = [
        SearchResult("guide", index, f"chunk {index}", 1.0 - index / 10) for index in range(5)
    ]
    service = RagService(provider, FakeRepository(matches), chunk_size=100, chunk_overlap=20)

    _answer, sources = await service.ask("question", top_k=2)

    assert len(sources) == 2
    assert provider.answer_contexts == ["chunk 0", "chunk 1"]


@pytest.mark.asyncio
async def test_ask_without_indexed_documents_is_explicit() -> None:
    service = RagService(FakeProvider(), FakeRepository(), chunk_size=100, chunk_overlap=20)

    with pytest.raises(NoDocumentsError):
        await service.ask("anything?", top_k=3)
