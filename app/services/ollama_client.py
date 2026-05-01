from typing import Any

import httpx


class OllamaClientError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": model,
            "input": texts,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaClientError("Ollama embedding request failed.") from exc

        data = response.json()
        return self._parse_embeddings(data)

    def generate(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaClientError("Ollama generation request failed.") from exc

        data = response.json()
        return self._parse_generation(data)

    def _parse_embeddings(self, data: dict[str, Any]) -> list[list[float]]:
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise OllamaClientError("Ollama response does not contain embeddings.")

        parsed_embeddings: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise OllamaClientError("Ollama returned an invalid embedding.")
            parsed_embeddings.append([float(value) for value in embedding])

        return parsed_embeddings

    def _parse_generation(self, data: dict[str, Any]) -> str:
        generated_text = data.get("response")

        if not isinstance(generated_text, str):
            raise OllamaClientError("Ollama response does not contain generated text.")

        return generated_text.strip()
