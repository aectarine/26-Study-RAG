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
            # 같은 본문은 거리가 가장 가까운 것만 남깁니다.
            unique = {}
            for document in documents:
                unique.setdefault(document["content"].strip(), document)
            documents = list(unique.values())[:SEARCH_LIMIT]
        if strategy == "semantic-deduplicated":
            # SQL은 후보를 CANDIDATE_LIMIT까지 남기므로 여기서 SEARCH_LIMIT으로 자릅니다.
            candidates = documents
            normalized = [" ".join(d["content"].split()) for d in candidates]
            documents = [d for i, d in enumerate(candidates) if not any(
                i != j and len(normalized[i]) > len(normalized[j])
                and normalized[j] in normalized[i]
                and candidates[j]["distance"] <= d["distance"]
                for j in range(len(candidates))
            )][:SEARCH_LIMIT]
        return documents
