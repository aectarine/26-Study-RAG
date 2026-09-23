from fastapi import APIRouter, Depends
from fastapi_utils.cbv import cbv

from core.dep.container import get_rag_service
from core.schema.request import SearchRequest
from service.rag_service import RagService


router = APIRouter(tags=["검색 API"])
test_router = APIRouter(tags=["테스트 API"])


@cbv(router)
class SearchRouter:
    service: RagService = Depends(get_rag_service)

    @router.post("/")
    async def find_documents_by_question(self, request: SearchRequest):
        documents = await self.service.find_documents_by_question(request.question)
        return {"question": request.question, "documents": documents}


@cbv(test_router)
class SearchTestRouter:
    service: RagService = Depends(get_rag_service)

    @test_router.post("/filtered")
    async def find_filtered_documents_by_question(self, request: SearchRequest):
        documents = await self.service.find_documents_by_question(
            request.question, "filtered"
        )
        return {"question": request.question, "documents": documents}
