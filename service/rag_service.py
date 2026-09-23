import time

from sqlalchemy.ext.asyncio import AsyncSession

from core.embed.embedding import EmbeddingClient
from repo.document_repo import DocumentRepository


SEARCH_LIMIT = 3
CANDIDATE_LIMIT = 5
DISTANCE_THRESHOLD = 0.5
DUPLICATE_DISTANCE_THRESHOLD = 0.1


class RagService:
    def __init__(self, session: AsyncSession, repository: DocumentRepository,
                 embedding_client: EmbeddingClient):
        self._session = session
        self._repository = repository
        self.embedding_client = embedding_client

    async def find_documents_by_question(self, question,
                                         strategy="basic", timings=None):
        embedding = await self.embedding_client.create_embedding(question, timings)
        started = time.perf_counter()
        async with self._session.begin():
            rows = await self._repository.find_rag_documents_by_embedding(
                embedding,
                strategy,
                SEARCH_LIMIT,
                DISTANCE_THRESHOLD,
                CANDIDATE_LIMIT,
                DUPLICATE_DISTANCE_THRESHOLD,
            )
        if timings is not None:
            timings["db-time"] = time.perf_counter() - started
        documents = [
            {"id": row[0], "content": row[1], "distance": float(row[2]),
             "source_document_id": row[3], "filename": row[4]}
            for row in rows
        ]
        if strategy == "deduplicated":
            seen = set()
            documents = [d for d in documents if not (d["content"].strip() in seen or seen.add(d["content"].strip()))]
        if strategy == "semantic-deduplicated":
            normalized = [" ".join(d["content"].split()) for d in documents]
            documents = [d for i, d in enumerate(documents) if not any(
                i != j and len(normalized[i]) > len(normalized[j])
                and normalized[j] in normalized[i]
                and documents[j]["distance"] <= d["distance"]
                for j in range(len(documents))
            )]
        return documents
