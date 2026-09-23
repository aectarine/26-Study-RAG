from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.embed.embedding import EmbeddingClient
from core.util.document import create_content_hash, split_text
from repo.document_repo import DocumentRepository
from service.chunk_service import ChunkService


class DocumentService:
    """문서 업무의 트랜잭션 경계. ChunkService와 동일한 세션을 사용합니다."""

    def __init__(self, session: AsyncSession, repository: DocumentRepository,
                 chunk_service: ChunkService, embedding_client: EmbeddingClient):
        self._session = session
        self._repository = repository
        self._chunk_service = chunk_service
        self._embedding_client = embedding_client

    async def count_documents(self) -> int:
        async with self._session.begin():
            return await self._repository.count_rag_documents()

    async def find_all_documents(self) -> list[dict]:
        async with self._session.begin():
            rows = await self._repository.find_all_source_documents()
            return [
                {"source_document_id": row[0], "filename": row[1],
                 "inserted": row[2], "chunk_count": row[3]}
                for row in rows
            ]

    async def find_document_by_id(self, source_document_id: int) -> dict:
        async with self._session.begin():
            source = await self._repository.find_source_document_by_id(source_document_id)
            if source is None:
                raise HTTPException(404, "해당 문서를 찾을 수 없습니다.")
            chunks = await self._repository.find_rag_documents_by_source_document_id(
                source_document_id
            )
            return {
                "source_document_id": source.id, "filename": source.filename,
                "content_hash": source.content_hash, "inserted": source.inserted,
                "chunk_count": len(chunks),
                "chunks": [{"chunk_id": row[0], "content": row[1]} for row in chunks],
            }

    async def create_document(self, content: str):
        embedding = await self._embedding_client.create_embedding(content)
        async with self._session.begin():
            chunk = await self._repository.save_rag_document(None, content, embedding)
            rs = {"document_id": chunk.id, "content": chunk.content,
                  "embedding_dimension": len(embedding)}
        return rs

    async def _find_duplicate(self, filename: str, content_hash: str):
        """현재 트랜잭션 안에서 검사하며 자체 트랜잭션은 시작하지 않습니다."""
        existing = await self._repository.find_source_document_by_filename_or_hash(
            filename, content_hash
        )
        if existing is None:
            return None
        if existing.filename != filename:
            raise HTTPException(
                409, f"같은 내용이 '{existing.filename}'으로 이미 등록되어 있습니다."
            )
        if existing.content_hash != content_hash:
            raise HTTPException(409, "같은 파일명으로 등록된 문서가 있지만 내용이 다릅니다.")
        return {"filename": filename, "source_document_id": existing.id,
                "status": "skipped", "message": "이미 등록된 동일한 문서입니다."}

    async def upload_document(self, filename: str, content: str):
        chunks = split_text(content)
        content_hash = create_content_hash(content)
        async with self._session.begin():
            duplicate = await self._find_duplicate(filename, content_hash)
            if duplicate:
                return duplicate

        # 느린 외부 호출 중 DB 연결을 점유하지 않습니다.
        embeddings = [
            await self._embedding_client.create_embedding(chunk) for chunk in chunks
        ]
        try:
            async with self._session.begin():
                # 임베딩 생성 중 등록된 문서가 있는지 다시 검사합니다.
                duplicate = await self._find_duplicate(filename, content_hash)
                if duplicate:
                    return duplicate
                source = await self._repository.save_source_document(filename, content_hash)
                documents = await self._chunk_service.save_chunks(source.id, chunks, embeddings)
                rs = {"filename": filename, "source_document_id": source.id,
                      "chunk_count": len(documents), "documents": documents, "status": "saved"}
        except IntegrityError as error:
            # 동시 요청이 먼저 저장한 경우이며, 재검사가 충돌 원인을 알려 줍니다.
            async with self._session.begin():
                duplicate = await self._find_duplicate(filename, content_hash)
                if duplicate:
                    return duplicate
            raise HTTPException(409, "이미 등록된 문서와 충돌했습니다.") from error
        return rs

    async def replace_document_by_id(
            self, source_document_id: int, filename: str,
            content: str, force: bool = False):
        chunks = split_text(content)
        new_hash = create_content_hash(content)
        async with self._session.begin():
            source = await self._repository.find_source_document_by_id(source_document_id)
            if source is None:
                raise HTTPException(404, "해당 문서를 찾을 수 없습니다.")
            if source.content_hash == new_hash and not force:
                return {"source_document_id": source_document_id, "status": "skipped",
                        "message": "기존 문서와 내용이 동일합니다. force=true를 사용하세요."}

        embeddings = [
            await self._embedding_client.create_embedding(chunk) for chunk in chunks
        ]

        try:
            async with self._session.begin():
                # 같은 문서의 교체/삭제 작업을 직렬화하고 최신 값을 다시 읽습니다.
                source = await self._repository.find_source_document_by_id(
                    source_document_id, for_update=True
                )
                if source is None:
                    raise HTTPException(404, "해당 문서를 찾을 수 없습니다.")
                if source.content_hash == new_hash and not force:
                    return {"source_document_id": source_document_id, "status": "skipped",
                            "message": "기존 문서와 내용이 동일합니다. force=true를 사용하세요."}
                await self._chunk_service.delete_chunks_by_source_id(source_document_id)
                documents = await self._chunk_service.save_chunks(
                    source_document_id, chunks, embeddings
                )
                source.filename = filename
                source.content_hash = new_hash
        except IntegrityError as error:
            # 다른 문서가 이미 같은 파일명을 사용 중입니다.
            raise HTTPException(
                409, "같은 파일명으로 등록된 다른 문서가 있습니다."
            ) from error

        return {"filename": filename, "source_document_id": source_document_id,
                "chunk_count": len(documents), "documents": documents, "status": "replaced"}

    async def delete_document_by_id(self, source_document_id: int):
        async with self._session.begin():
            source = await self._repository.find_source_document_by_id(
                source_document_id, for_update=True
            )
            if source is None:
                raise HTTPException(404, "해당 문서를 찾을 수 없습니다.")
            filename = source.filename
            count = await self._chunk_service.delete_chunks_by_source_id(source_document_id)
            await self._repository.delete_source_document(source)
        return {"source_document_id": source_document_id, "filename": filename,
                "deleted_chunk_count": count, "status": "deleted"}
