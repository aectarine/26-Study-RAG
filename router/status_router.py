from fastapi import APIRouter, Depends
from fastapi_utils.cbv import cbv

from core.dep.container import get_document_service, get_embedding_client
from core.embed.embedding import EmbeddingClient
from core.schema.request import EmbeddingRequest
from service.document_service import DocumentService


router = APIRouter(tags=["상태 확인"])
test_router = APIRouter(tags=["테스트 API"])
db_test_router = APIRouter()
embedding_test_router = APIRouter()


@cbv(router)
class StatusRouter:
    @router.get("/")
    async def get_root_status(self):
        return {"message": "Study-RAG Server Running"}

    @router.get("/health")
    async def get_health_status(self):
        return {"status": "ok"}


@cbv(db_test_router)
class DatabaseTestRouter:
    service: DocumentService = Depends(get_document_service)

    @db_test_router.get("/db-test")
    async def test_database_connection(self):
        count = await self.service.count_documents()
        return {"status": "connected", "document_count": count}


@cbv(embedding_test_router)
class EmbeddingTestRouter:
    client: EmbeddingClient = Depends(get_embedding_client)

    @embedding_test_router.post("/embedding-test")
    async def test_embedding_generation(self, request: EmbeddingRequest):
        embedding = await self.client.create_embedding(request.text)
        return {"text": request.text, "dimension": len(embedding),
                "embedding_preview": embedding[:5]}


test_router.include_router(db_test_router)
test_router.include_router(embedding_test_router)
