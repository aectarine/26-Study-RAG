"""외부 서버 없이 DI 대체를 통해 오류 응답 계약을 검증합니다."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from core.dep.container import get_document_service, get_embedding_client


async def run_fault_api_tests() -> list[str]:
    rs = []
    service = SimpleNamespace(
        count_documents=AsyncMock(side_effect=SQLAlchemyError("mock database failure"))
    )
    embedding = SimpleNamespace(create_embedding=AsyncMock())
    previous = main.app.dependency_overrides.copy()
    main.app.dependency_overrides[get_document_service] = lambda: service
    main.app.dependency_overrides[get_embedding_client] = lambda: embedding
    try:
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/test/db-test")
            assert response.status_code == 503, response.text
            assert response.json()["error"] == "database_unavailable"
            rs.append("DB 오류 → 503")
            for error, status, error_code in [
                (httpx.ConnectError("mock connection"), 503, "ollama_unavailable"),
                (httpx.ReadTimeout("mock timeout"), 504, "ollama_timeout"),
                (httpx.PoolTimeout("mock pool"), 503, "ollama_busy"),
            ]:
                embedding.create_embedding.side_effect = error
                response = await client.post("/api/test/embedding-test", json={"text": "test"})
                assert response.status_code == status, response.text
                assert response.json()["error"] == error_code
                rs.append(f"{error_code} → {status}")
    finally:
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(previous)
    return rs


if __name__ == "__main__":
    print("[PASS] " + ", ".join(asyncio.run(run_fault_api_tests())))
