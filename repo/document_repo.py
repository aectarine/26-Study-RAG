from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.model.entity import RagDocument, SourceDocument


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def count_rag_documents(self) -> int:
        rs = await self._session.execute(
            select(func.count()).select_from(RagDocument),
        )
        return int(rs.scalar_one())

    async def find_source_document_by_filename_or_hash(
            self, filename: str, content_hash: str):
        """파일명 또는 내용 해시가 겹치는 문서를 찾으며 파일명 일치를 우선합니다."""
        rs = await self._session.execute(
            select(SourceDocument)
            .where(or_(SourceDocument.filename == filename,
                       SourceDocument.content_hash == content_hash))
            .order_by((SourceDocument.filename == filename).desc(),
                      SourceDocument.id.desc())
            .limit(1)
        )
        return rs.scalar_one_or_none()

    async def find_source_document_by_id(self, source_id: int, for_update: bool = False):
        return await self._session.get(
            SourceDocument, source_id,
            with_for_update=for_update, populate_existing=for_update,
        )

    async def save_source_document(self, filename: str, content_hash: str):
        source = SourceDocument(filename=filename, content_hash=content_hash)
        self._session.add(source)
        await self._session.flush()
        return source

    async def save_rag_document(self, source_id: int | None, content: str, embedding):
        chunk = RagDocument(
            source_document_id=source_id,
            content=content,
            embedding=embedding,
        )
        self._session.add(chunk)
        await self._session.flush()
        return chunk

    async def find_all_source_documents(self):
        rs = await self._session.execute(
            select(
                SourceDocument.id,
                SourceDocument.filename,
                SourceDocument.inserted,
                func.count(RagDocument.id),
            )
            .outerjoin(RagDocument)
            .group_by(SourceDocument.id)
            .order_by(SourceDocument.id.desc())
        )
        return rs.all()

    async def find_rag_documents_by_source_document_id(self, source_id: int):
        rs = await self._session.execute(
            select(RagDocument.id, RagDocument.content)
            .where(RagDocument.source_document_id == source_id)
            .order_by(RagDocument.id)
        )
        return rs.all()

    async def delete_rag_documents_by_source_document_id(self, source_id: int):
        rs = await self._session.execute(
            delete(RagDocument).where(RagDocument.source_document_id == source_id),
        )
        return int(rs.rowcount or 0)

    async def delete_source_document(self, source: SourceDocument):
        await self._session.delete(source)

    async def find_rag_documents_by_embedding(
            self,
            embedding: list[float], strategy: str, limit: int,
            distance_threshold: float, candidate_limit: int,
            duplicate_threshold: float):
        embedding_value = str(embedding)
        if strategy == "semantic-deduplicated":
            query = text("""
                         WITH candidates AS (SELECT r.id,
                                                    r.content,
                                                    r.embedding,
                                                    r.embedding <=> CAST(:embedding AS vector) AS distance,
                                                    r.source_document_id,
                                                    s.filename
                                             FROM rag_documents r
                                                      LEFT JOIN source_documents s ON r.source_document_id = s.id
                                             ORDER BY distance
                                             LIMIT :candidate_limit),
                              duplicate_pairs AS (SELECT a.id       first_id,
                                                         b.id       second_id,
                                                         a.distance first_distance,
                                                         b.distance second_distance
                                                  FROM candidates a
                                                           CROSS JOIN candidates b
                                                  WHERE a.id < b.id
                                                    AND a.embedding <=> b.embedding <= :duplicate_threshold),
                              removed_documents AS (SELECT CASE
                                                               WHEN first_distance <= second_distance
                                                                   THEN second_id
                                                               ELSE first_id END remove_id
                                                    FROM duplicate_pairs)
                         SELECT id, content, distance, source_document_id, filename
                         FROM candidates
                         WHERE id NOT IN (SELECT remove_id FROM removed_documents)
                         ORDER BY distance
                         LIMIT :candidate_limit
                         """)
            params = {"embedding": embedding_value, "candidate_limit": candidate_limit,
                      "duplicate_threshold": duplicate_threshold}
        elif strategy == "deduplicated-db":
            query = text("""
                         WITH candidates AS (SELECT r.id,
                                                    r.content,
                                                    r.embedding <=> CAST(:embedding AS vector) AS distance,
                                                    r.source_document_id,
                                                    s.filename
                                             FROM rag_documents r
                                                      LEFT JOIN source_documents s ON r.source_document_id = s.id
                                             ORDER BY distance
                                             LIMIT :candidate_limit),
                              unique_contents AS (SELECT DISTINCT ON (content) id,
                                                                               content,
                                                                               distance,
                                                                               source_document_id,
                                                                               filename
                                                  FROM candidates
                                                  ORDER BY content, distance, id)
                         SELECT id, content, distance, source_document_id, filename
                         FROM unique_contents
                         ORDER BY distance, id
                         LIMIT :limit
                         """)
            params = {"embedding": embedding_value, "candidate_limit": candidate_limit, "limit": limit}
        else:
            filter_sql = ""
            if strategy == "filtered":
                filter_sql = "WHERE r.embedding <=> CAST(:embedding AS vector) <= :threshold"
            query = text(f"""
                SELECT r.id, r.content,
                       r.embedding <=> CAST(:embedding AS vector) AS distance,
                       r.source_document_id, s.filename
                FROM rag_documents r
                LEFT JOIN source_documents s ON r.source_document_id = s.id
                {filter_sql}
                ORDER BY distance LIMIT :limit
            """)
            # 중복 제거로 결과가 줄지 않도록 후보를 넉넉히 조회합니다.
            params = {"embedding": embedding_value, "threshold": distance_threshold,
                      "limit": candidate_limit if strategy == "deduplicated" else limit}
        rs = await self._session.execute(query, params)
        return rs.all()
