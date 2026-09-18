import pytest

from rag_app.chunking import chunk_text


def test_short_text_becomes_one_normalized_chunk() -> None:
    assert chunk_text("  hello\n\nworld  ") == ["hello world"]


def test_long_text_creates_overlapping_chunks() -> None:
    chunks = chunk_text(" ".join(f"word-{index}" for index in range(80)), size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_invalid_chunk_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("content", size=size, overlap=overlap)
