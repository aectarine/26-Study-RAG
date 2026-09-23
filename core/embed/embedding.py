import time

import httpx

from core.cfg.settings import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self.settings = settings
        self._http_client = http_client

    async def create_embedding(
            self, text: str,
            timings: dict[str, float] | None = None) -> list[float]:
        started = time.perf_counter()
        response = await self._http_client.post(
            f"{self.settings.ollama_url}/api/embed",
            timeout=httpx.Timeout(self.settings.embedding_timeout,
                                  pool=self.settings.http_pool_timeout),
            json={"model": self.settings.embedding_model, "input": text},
        )
        response.raise_for_status()
        rs = response.json()
        if timings is not None:
            timings["embed-time"] = time.perf_counter() - started
        return rs["embeddings"][0]
