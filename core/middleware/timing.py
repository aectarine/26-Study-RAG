import logging
import time

from fastapi import FastAPI, Request


logger = logging.getLogger(__name__)


def register_timing_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def measure_response_time(request: Request, call_next):
        request.state.timings = {}
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        header_names = {
            "embed-time": "X-Embedding-Time",
            "db-time": "X-DB-Time",
            "rerank-time": "X-Rerank-Time",
            "answer-time": "X-Answer-Time",
        }
        for name, value in request.state.timings.items():
            if name in header_names:
                response.headers[header_names[name]] = f"{value:.3f}"
        logger.info(
            "request method=%s path=%s status=%s process_time=%.3fs",
            request.method, request.url.path, response.status_code, elapsed
        )
        return response
