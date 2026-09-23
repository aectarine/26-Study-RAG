"""외부 DB/Ollama 없이 요청 범위 DI와 실제 SQLAlchemy 트랜잭션을 검증합니다."""
import asyncio
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from dependency_injector import providers
from fastapi import Depends, FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from core.db.database import Database
from core.dep.container import container, get_document_service, get_rag_service
from core.model.entity import Base, RagDocument, SourceDocument
from repo.document_repo import DocumentRepository
from service.chunk_service import ChunkService
from service.document_service import DocumentService
from service.rag_service import RagService


class ArchitectureTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.sessions = []
        sessions = self.sessions

        class TrackedSession(AsyncSession):
            closed = False

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                sessions.append(self)

            async def close(self):
                await super().close()
                self.closed = True

        self.factory = async_sessionmaker(
            self.engine, class_=TrackedSession, expire_on_commit=False
        )
        self.database = SimpleNamespace(
            session_factory=self.factory, close=AsyncMock(side_effect=self.engine.dispose)
        )

        async def embed(*args, **kwargs):
            self.assertTrue(all(not session.in_transaction() for session in self.sessions))
            return [0.1] * 768

        async def answer(*args, **kwargs):
            self.assertTrue(all(not session.in_transaction() for session in self.sessions))
            return "test answer"

        self.embedding = SimpleNamespace(create_embedding=AsyncMock(side_effect=embed))
        self.ollama = SimpleNamespace(
            generate_answer=AsyncMock(side_effect=answer),
            rerank_documents=AsyncMock(side_effect=lambda question, documents, timings: documents),
        )
        self.stack = ExitStack()
        self.stack.enter_context(container.database.override(providers.Object(self.database)))
        self.stack.enter_context(container.embedding_client.override(providers.Object(self.embedding)))
        self.stack.enter_context(container.ollama_client.override(providers.Object(self.ollama)))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
            base_url="http://testserver", follow_redirects=False,
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.stack.close()
        await self.engine.dispose()

    async def upload(self):
        return await self.client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", "first paragraph\n\nsecond paragraph", "text/plain")},
        )

    async def counts(self):
        async with self.factory() as session:
            return (
                await session.scalar(select(func.count()).select_from(SourceDocument)),
                await session.scalar(select(func.count()).select_from(RagDocument)),
            )

    async def test_document_routes_commit_and_cleanup(self):
        response = await self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        source_id = response.json()["source_document_id"]
        self.assertTrue(all(session.closed for session in self.sessions))
        self.assertEqual((await self.counts())[0], 1)

        embedding_calls = self.embedding.create_embedding.await_count
        duplicate = await self.upload()
        self.assertEqual(duplicate.json()["status"], "skipped")
        self.assertEqual(self.embedding.create_embedding.await_count, embedding_calls)
        response = await self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        response = await self.client.get(f"/api/documents/{source_id}")
        self.assertEqual(response.json()["filename"], "test.txt")
        response = await self.client.put(
            f"/api/documents/{source_id}",
            files={"file": ("new.txt", "replacement content", "text/plain")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "replaced")
        response = await self.client.delete(f"/api/documents/{source_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.counts(), (0, 0))
        self.assertEqual((await self.client.get(f"/api/documents/{source_id}")).status_code, 404)
        response = await self.client.post("/api/documents", json={"content": "direct"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(session.closed for session in self.sessions))

    async def test_child_service_failure_rolls_back_parent_and_chunks(self):
        class FailingChunkService(ChunkService):
            async def save_chunks(self, *args, **kwargs):
                await super().save_chunks(*args, **kwargs)
                raise RuntimeError("failure after child writes")

        with container.chunk_service.override(providers.Factory(FailingChunkService)):
            response = await self.upload()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(await self.counts(), (0, 0))
        self.assertTrue(all(session.closed for session in self.sessions))

    async def test_failed_replacement_preserves_original(self):
        response = await self.upload()
        source_id = response.json()["source_document_id"]
        original = (await self.client.get(f"/api/documents/{source_id}")).json()

        class FailingChunkService(ChunkService):
            async def save_chunks(self, *args, **kwargs):
                await super().save_chunks(*args, **kwargs)
                raise RuntimeError("replacement failure")

        with container.chunk_service.override(providers.Factory(FailingChunkService)):
            response = await self.client.put(
                f"/api/documents/{source_id}",
                files={"file": ("changed.txt", "changed body", "text/plain")},
            )
        self.assertEqual(response.status_code, 500)
        restored = (await self.client.get(f"/api/documents/{source_id}")).json()
        self.assertEqual(original, restored)

    async def test_cancellation_rolls_back_and_closes_session(self):
        class CancelledChunkService(ChunkService):
            async def save_chunks(self, *args, **kwargs):
                await super().save_chunks(*args, **kwargs)
                raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            async with self.factory() as session:
                repo = container.document_repository(session=session)
                service = container.document_service(
                    session=session, repository=repo,
                    chunk_service=CancelledChunkService(repo),
                )
                await service.upload_document("cancel.txt", "cancelled")
        self.assertTrue(session.closed)
        self.assertFalse(session.in_transaction())
        self.assertEqual(await self.counts(), (0, 0))

    async def test_request_cache_and_concurrent_isolation(self):
        app = FastAPI()
        observed = []
        barrier = asyncio.Barrier(4)

        @app.get("/scope")
        async def scope(
                first: DocumentService = Depends(get_document_service),
                second: DocumentService = Depends(get_document_service),
                rag: RagService = Depends(get_rag_service)):
            self.assertIs(first, second)
            self.assertIs(first._session, rag._session)
            self.assertIs(first._repository, rag._repository)
            self.assertIs(first._repository, first._chunk_service._repository)
            observed.append(first)
            await barrier.wait()
            return {"ok": True}

        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.wait_for(
                asyncio.gather(*(client.get("/scope") for _ in range(4))), timeout=10
            )
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len({id(service) for service in observed}), 4)
        self.assertEqual(len({id(service._session) for service in observed}), 4)
        self.assertTrue(all(session.closed for session in self.sessions))

    async def test_chat_search_routes_and_connection_release(self):
        rows = [(1, "content", 0.2, 1, "test.txt")]
        with patch.object(
                DocumentRepository, "find_rag_documents_by_embedding",
                new=AsyncMock(return_value=rows)):
            paths = ["/api/chat"] + [
                "/api/test/chat/" + strategy for strategy in
                ("basic", "filtered", "reranked", "deduplicated", "deduplicated-db",
                 "semantic-deduplicated", "semantic-reranked")
            ]
            for path in paths:
                response = await self.client.post(path, json={"question": "test"})
                self.assertEqual(response.status_code, 200, (path, response.text))
                self.assertEqual(response.json()["answer"], "test answer")
                self.assertEqual(response.json()["sources"][0]["id"], 1)
            for path in ("/api/search", "/api/test/search/filtered"):
                response = await self.client.post(path, json={"question": "test"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["documents"][0]["id"], 1)
        self.assertTrue(all(session.closed for session in self.sessions))

    async def test_openapi_and_status(self):
        schema = main.app.openapi()
        self.assertEqual(len(schema["paths"]), 17)
        for path in ("/api/chat", "/api/search", "/api/documents"):
            self.assertIn(path, schema["paths"])
            self.assertNotIn(path + "/", schema["paths"])
        for path, methods in schema["paths"].items():
            for operation in methods.values():
                names = {param["name"] for param in operation.get("parameters", [])}
                self.assertFalse(names & {"self", "session", "service", "repository"}, path)
        for path in ("/api/", "/api/health"):
            self.assertEqual((await self.client.get(path)).status_code, 200)
        self.assertEqual(len(self.sessions), 0)
        self.assertEqual((await self.client.get("/api/test/db-test")).status_code, 200)

    async def test_lifespan_closes_shared_resources(self):
        client = httpx.AsyncClient()
        with container.http_client.override(providers.Object(client)):
            async with main.app.router.lifespan_context(main.app):
                self.assertFalse(client.is_closed)
            self.assertTrue(client.is_closed)
            self.database.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
