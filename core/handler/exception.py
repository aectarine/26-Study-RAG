import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(httpx.ConnectError)
    async def ollama_connection_error(request: Request, exc: httpx.ConnectError):
        logger.error("Ollama 연결 실패: %s", exc)
        return JSONResponse(status_code=503, content={
            "error": "ollama_unavailable",
            "message": "Ollama 서버에 연결할 수 없습니다.",
        })

    @app.exception_handler(httpx.TimeoutException)
    async def ollama_timeout_error(request: Request, exc: httpx.TimeoutException):
        logger.error("Ollama 요청 시간 초과: %s", exc)
        return JSONResponse(status_code=504, content={
            "error": "ollama_timeout",
            "message": "Ollama 답변 생성 시간이 초과되었습니다.",
        })

    @app.exception_handler(httpx.PoolTimeout)
    async def ollama_pool_timeout(request: Request, exc: httpx.PoolTimeout):
        return JSONResponse(status_code=503, content={
            "error": "ollama_busy",
            "message": "Ollama 요청이 많습니다. 잠시 후 다시 시도하세요.",
        })

    async def database_error(request: Request, exc: Exception):
        logger.error("데이터베이스 오류: %s", exc)
        return JSONResponse(status_code=503, content={
            "error": "database_unavailable",
            "message": "데이터베이스에 연결할 수 없습니다.",
        })

    app.add_exception_handler(SQLAlchemyError, database_error)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("처리되지 않은 서버 오류")
        return JSONResponse(status_code=500, content={
            "error": "internal_server_error",
            "message": "서버 내부 오류가 발생했습니다.",
        })
