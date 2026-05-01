from app.services.ollama_client import OllamaClient, OllamaClientError


class EmbeddingServiceError(Exception):
    pass


class EmbeddingService:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        expected_dimension: int,
    ) -> None:
        self.client = client
        self.model = model
        self.expected_dimension = expected_dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.client.embed(model=self.model, texts=texts)
        except OllamaClientError as exc:
            raise EmbeddingServiceError(f"Could not generate embeddings. {exc}") from exc

        if len(embeddings) != len(texts):
            raise EmbeddingServiceError("Embedding count does not match text count.")

        for embedding in embeddings:
            if len(embedding) != self.expected_dimension:
                raise EmbeddingServiceError(
                    f"Expected embedding dimension {self.expected_dimension}, "
                    f"got {len(embedding)}."
                )

        return embeddings
