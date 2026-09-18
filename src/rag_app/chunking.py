def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into retrieval-sized windows with a little shared context."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    # Normalizing whitespace prevents formatting-only differences from producing
    # different embeddings and keeps chunk boundaries deterministic.
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            # Prefer a word boundary without allowing one unusually long word to
            # shrink a chunk below half of the configured target size.
            boundary = normalized.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary

        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break

        # Overlap repeats the end of one chunk at the beginning of the next so a
        # fact spanning a boundary remains retrievable as a complete thought.
        start = max(end - overlap, start + 1)

    return chunks
