from collections.abc import AsyncIterator

import httpx
from dependency_injector import containers, providers
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.cfg.settings import get_settings
from core.db.database import Database
from core.embed.embedding import EmbeddingClient
from core.llm.ollama import OllamaClient
from repo.document_repo import DocumentRepository
from service.chat_service import ChatService
from service.chunk_service import ChunkService
from service.document_service import DocumentService
from service.rag_service import RagService


class Container(containers.DeclarativeContainer):
    """공유 자원은 Singleton, 업무 객체는 Factory로 조립합니다."""

    settings = providers.Singleton(get_settings)
    database = providers.Singleton(Database, settings=settings)
    http_client = providers.Singleton(
        httpx.AsyncClient,
        limits=providers.Factory(
            httpx.Limits,
            max_connections=settings.provided.http_max_connections,
            max_keepalive_connections=settings.provided.http_max_connections,
        ),
    )
    document_repository = providers.Factory(DocumentRepository)
    chunk_service = providers.Factory(ChunkService)
    embedding_client = providers.Singleton(
        EmbeddingClient, settings=settings, http_client=http_client
    )
    ollama_client = providers.Singleton(
        OllamaClient, settings=settings, http_client=http_client
    )
    rag_service = providers.Factory(
        RagService,
        embedding_client=embedding_client,
    )
    document_service = providers.Factory(
        DocumentService,
        embedding_client=embedding_client,
    )
    chat_service = providers.Factory(
        ChatService,
        ollama_client=ollama_client,
    )


container = Container()


# 요청별 조립: Depends 캐시로 같은 요청 안에서 객체와 세션을 재사용합니다.
async def get_session() -> AsyncIterator[AsyncSession]:
    async with container.database().session_factory() as session:
        yield session


async def get_document_repository(
        session: AsyncSession = Depends(get_session)) -> DocumentRepository:
    return container.document_repository(session=session)


async def get_chunk_service(
        repository: DocumentRepository = Depends(get_document_repository),
) -> ChunkService:
    return container.chunk_service(repository=repository)


async def get_document_service(
        session: AsyncSession = Depends(get_session),
        repository: DocumentRepository = Depends(get_document_repository),
        chunk_service: ChunkService = Depends(get_chunk_service),
) -> DocumentService:
    return container.document_service(
        session=session, repository=repository, chunk_service=chunk_service
    )


async def get_rag_service(
        session: AsyncSession = Depends(get_session),
        repository: DocumentRepository = Depends(get_document_repository),
) -> RagService:
    return container.rag_service(session=session, repository=repository)


async def get_chat_service(
        rag_service: RagService = Depends(get_rag_service)) -> ChatService:
    return container.chat_service(rag_service=rag_service)


async def get_embedding_client() -> EmbeddingClient:
    return container.embedding_client()
