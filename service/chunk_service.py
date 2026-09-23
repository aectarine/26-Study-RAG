from repo.document_repo import DocumentRepository


class ChunkService:
    """호출한 업무의 트랜잭션에 참여합니다. begin/commit은 호출하지 않습니다."""

    def __init__(self, repository: DocumentRepository):
        self._repository = repository

    async def save_chunks(self, source_id: int, chunks, embeddings) -> list[dict]:
        documents = []
        for content, embedding in zip(chunks, embeddings, strict=True):
            saved = await self._repository.save_rag_document(source_id, content, embedding)
            documents.append({"document_id": saved.id, "content": saved.content})
        return documents

    async def delete_chunks_by_source_id(self, source_id: int) -> int:
        return await self._repository.delete_rag_documents_by_source_document_id(source_id)
