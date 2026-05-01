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

        response = self._post_json("/api/embed", payload, "Ollama embedding request")
        data = response.json()
        return self._parse_embeddings(data)

    def generate(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        response = self._post_json("/api/generate", payload, "Ollama generation request")
        data = response.json()
        return self._parse_generation(data)

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        error_label: str,
    ) -> httpx.Response:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                self._format_status_error(error_label, exc.response)
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaClientError(
                f"{error_label} timed out after {self.timeout_seconds} seconds."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"{error_label} failed: {exc}") from exc

        return response

    def _format_status_error(self, error_label: str, response: httpx.Response) -> str:
        body = response.text.strip()
        if len(body) > 300:
            body = f"{body[:300]}..."

        reason = body or response.reason_phrase
        return f"{error_label} failed with status {response.status_code}: {reason}"

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
