import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    metadata: dict[str, int]


class ChunkingService:
    def __init__(self, chunk_size_chars: int, chunk_overlap_chars: int) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be greater than 0.")
        if chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars cannot be negative.")
        if chunk_overlap_chars >= chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars.")

        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def split_text(self, text: str) -> list[TextChunk]:
        words = re.findall(r"\S+", text.strip())
        if not words:
            return []

        chunks: list[TextChunk] = []
        word_start = 0

        while word_start < len(words):
            word_end = self._find_chunk_end(words, word_start)
            chunk_words = words[word_start:word_end]
            content = " ".join(chunk_words)

            chunks.append(
                TextChunk(
                    index=len(chunks),
                    content=content,
                    metadata={
                        "word_start": word_start,
                        "word_end": word_end,
                        "char_count": len(content),
                    },
                )
            )

            if word_end >= len(words):
                break

            next_start = self._find_next_start(words, word_start, word_end)
            word_start = next_start

        return chunks

    def _find_chunk_end(self, words: list[str], word_start: int) -> int:
        char_count = 0
        word_end = word_start

        while word_end < len(words):
            word = words[word_end]
            separator_length = 1 if word_end > word_start else 0
            next_size = char_count + separator_length + len(word)

            if word_end > word_start and next_size > self.chunk_size_chars:
                break

            char_count = next_size
            word_end += 1

        return word_end

    def _find_next_start(self, words: list[str], word_start: int, word_end: int) -> int:
        overlap_start = word_end
        overlap_chars = 0

        while overlap_start > word_start:
            candidate_start = overlap_start - 1
            separator_length = 1 if overlap_chars > 0 else 0
            candidate_size = overlap_chars + separator_length + len(words[candidate_start])

            if overlap_chars > 0 and candidate_size > self.chunk_overlap_chars:
                break

            overlap_start = candidate_start
            overlap_chars = candidate_size

            if overlap_chars >= self.chunk_overlap_chars:
                break

        if overlap_start <= word_start:
            return word_end

        return overlap_start
