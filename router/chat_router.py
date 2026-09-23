from fastapi import APIRouter, Depends, Request
from fastapi_utils.cbv import cbv

from core.dep.container import get_chat_service
from core.schema.request import SearchRequest
from core.schema.response import ChatResponse
from service.chat_service import ChatService

router = APIRouter(tags=["운영 API"])
test_router = APIRouter(tags=["테스트 API"])


@cbv(router)
class ChatRouter:
    service: ChatService = Depends(get_chat_service)

    @router.post("/", response_model=ChatResponse)
    async def chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "semantic-deduplicated", request.state.timings
        )


@cbv(test_router)
class ChatTestRouter:
    service: ChatService = Depends(get_chat_service)

    @test_router.post("/basic", response_model=ChatResponse)
    async def basic_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "basic", request.state.timings
        )

    @test_router.post("/filtered", response_model=ChatResponse)
    async def filtered_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "filtered", request.state.timings
        )

    @test_router.post("/reranked", response_model=ChatResponse)
    async def reranked_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_reranked_answer_by_question(
            body.question, request.state.timings, semantic=False
        )

    @test_router.post("/deduplicated", response_model=ChatResponse)
    async def deduplicated_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "deduplicated", request.state.timings
        )

    @test_router.post("/deduplicated-db", response_model=ChatResponse)
    async def deduplicated_db_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "deduplicated-db", request.state.timings
        )

    @test_router.post("/semantic-deduplicated", response_model=ChatResponse)
    async def semantic_deduplicated_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_answer_by_question(
            body.question, "semantic-deduplicated", request.state.timings
        )

    @test_router.post("/semantic-reranked", response_model=ChatResponse)
    async def semantic_reranked_chat_by_question(self, request: Request, body: SearchRequest):
        return await self.service.generate_reranked_answer_by_question(
            body.question, request.state.timings, semantic=True
        )
