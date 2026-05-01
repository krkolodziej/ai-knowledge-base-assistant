import pytest

from app.services.chunking_service import ChunkingService


def test_chunking_splits_text_into_overlapping_chunks() -> None:
    service = ChunkingService(chunk_size_chars=20, chunk_overlap_chars=6)

    chunks = service.split_text("alpha beta gamma delta epsilon zeta")

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.content for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta epsilon",
        "epsilon zeta",
    ]
    assert chunks[1].metadata["word_start"] < chunks[0].metadata["word_end"]


def test_chunking_returns_empty_list_for_blank_text() -> None:
    service = ChunkingService(chunk_size_chars=20, chunk_overlap_chars=6)

    assert service.split_text("   ") == []


def test_chunking_rejects_overlap_greater_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        ChunkingService(chunk_size_chars=100, chunk_overlap_chars=100)
