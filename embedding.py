import httpx

OLLAMA_URL = "http://localhost:11434"

EMBEDDING_MODEL = "embeddinggemma"


async def create_embedding(text: str) -> list[float]:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    """

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            }
        )

        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
