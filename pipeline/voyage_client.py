"""Hosted Voyage embeddings; independent of the answer-generation provider."""
import httpx
from pydantic import SecretStr

EMBEDDING_MODELS = {
    "Voyage AI": ["voyage-4-lite", "voyage-4", "voyage-4-large", "voyage-code-4"],
    "Gemini": ["gemini-embedding-001", "gemini-embedding-2"],
    "Ollama": ["nomic-embed-text", "mxbai-embed-large"],
}


class VoyageClient:
    def __init__(self, api_key, timeout=120):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set VOYAGE_API_KEY in your environment or .streamlit/secrets.toml before using Voyage AI.")
        self.api_key = SecretStr(api_key.strip())
        self.timeout = timeout

    def _embed(self, texts, model, input_type):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Select a Voyage embedding model.")
        vectors = []
        with httpx.Client(timeout=self.timeout) as client:
            for offset in range(0, len(texts), 32):
                batch = texts[offset:offset + 32]
                position = f" Text chunks {offset + 1}–{offset + len(batch)} of {len(texts)} failed."
                try:
                    response = client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                        json={"input": batch, "model": model.strip(), "input_type": input_type,
                              "truncation": False, "output_dtype": "float"},
                    )
                    response.raise_for_status()
                    data = response.json()["data"]
                    if len(data) != len(batch) or sorted(row["index"] for row in data) != list(range(len(batch))):
                        raise ValueError("Invalid response indices")
                    vectors.extend(row["embedding"] for row in sorted(data, key=lambda row: row["index"]))
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    message = {
                        401: "Voyage rejected the API key. Check VOYAGE_API_KEY.",
                        403: "Voyage denied access. Check your key and model permissions.",
                        429: "Voyage quota or rate limit reached. Check your Voyage dashboard and retry later.",
                        400: "Voyage rejected the model or input. Check the selected model and input limits.",
                        404: "Voyage model or endpoint was not found.",
                    }.get(code, "Voyage service request failed. Retry later.")
                    raise ValueError(message + position) from None
                except httpx.HTTPError:
                    raise ValueError("Could not reach Voyage AI or the request timed out." + position) from None
                except (ValueError, KeyError, TypeError):
                    raise ValueError("Voyage returned an invalid embedding response." + position) from None
        return vectors

    def embed(self, texts, model):
        return self._embed(texts, model, "document")

    def embed_query(self, query, model):
        return self._embed([query], model, "query")[0]
