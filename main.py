from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.dep.container import container
from core.handler.exception import register_exception_handlers
from core.middleware.timing import register_timing_middleware
from core.util.route import include_api_router

from router.chat_router import router as chat_router
from router.chat_router import test_router as chat_test_router
from router.document_router import router as document_router
from router.search_router import router as search_router
from router.search_router import test_router as search_test_router
from router.status_router import router as status_router
from router.status_router import test_router as status_test_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    database = container.database()
    client = container.http_client()
    try:
        yield
    finally:
        try:
            await client.aclose()
        finally:
            try:
                await database.close()
            finally:
                container.reset_singletons()


app = FastAPI(
    lifespan=lifespan,
    title="Study-RAG",
    description="FastAPI + PostgreSQL + Ollama RAG 학습 프로젝트",
    version="2.0.0",
    openapi_tags=[
        {"name": "운영 API", "description": "검증된 검색 전략을 사용하는 실제 서비스 API"},
        {"name": "테스트 API", "description": "검색 전략 비교와 성능 측정을 위한 실험 API"},
        {"name": "검색 API", "description": "답변 생성 없이 업로드 문서 검색만 수행하는 API"},
        {"name": "업로드 문서 관리", "description": "업로드 문서 업로드, 조회, 수정, 삭제 API"},
        {"name": "상태 확인", "description": "서버와 DB 상태를 확인하는 API"},
    ],
)


register_timing_middleware(app)
register_exception_handlers(app)


API_PREFIX = "/api"
TEST_PREFIX = f"{API_PREFIX}/test"

app.include_router(status_router, prefix=API_PREFIX)
app.include_router(status_test_router, prefix=TEST_PREFIX)
include_api_router(app, search_router, prefix=f"{API_PREFIX}/search")
app.include_router(search_test_router, prefix=f"{TEST_PREFIX}/search")
include_api_router(app, chat_router, prefix=f"{API_PREFIX}/chat")
app.include_router(chat_test_router, prefix=f"{TEST_PREFIX}/chat")
include_api_router(app, document_router, prefix=f"{API_PREFIX}/documents")
