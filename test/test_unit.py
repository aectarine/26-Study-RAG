"""외부 서버 없이 문서 분할·설정·응답 경계값을 검증합니다."""
from io import BytesIO
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi import HTTPException, UploadFile

from core.cfg.settings import Settings
from core.llm.ollama import OllamaClient
from core.util.document import MAX_UPLOAD_BYTES, read_text_upload, split_text


class DocumentUnitTest(unittest.IsolatedAsyncioTestCase):
    def test_unbroken_text_preserves_every_character(self):
        content = "".join(chr(0x4E00 + index) for index in range(1600))
        chunks = split_text(content)
        self.assertEqual(chunks, [content[:700], content[600:1300], content[1200:]])

    def test_zero_overlap_preserves_tail(self):
        content = "가" * 700 + "나" * 700 + "다" * 10
        self.assertEqual("".join(split_text(content, overlap=0)), content)

    def test_invalid_chunk_options(self):
        for size, overlap in [(0, 0), (-1, 0), (700, -1), (10, 10)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                split_text("text", size, overlap)

    def test_database_url_preserves_credentials(self):
        settings = Settings()
        settings.db_password = "p@ss:/?#%word"
        self.assertEqual(settings.database_url.password, settings.db_password)
        self.assertNotIn(settings.db_password, str(settings.database_url))

    async def test_upload_read_is_bounded(self):
        file = UploadFile(filename="large.txt", file=BytesIO())
        file.read = AsyncMock(return_value=b"a" * (MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(HTTPException) as error:
            await read_text_upload(file)
        self.assertEqual(error.exception.status_code, 413)
        file.read.assert_awaited_once_with(MAX_UPLOAD_BYTES + 1)
        await file.close()

    async def test_upload_utf8_and_path_validation(self):
        file = UploadFile(filename="doc.txt", file=BytesIO("내용".encode("utf-8-sig")))
        self.assertEqual(await read_text_upload(file), "내용")
        await file.close()
        for filename in ("../doc.txt", "..\\doc.txt"):
            file = UploadFile(filename=filename, file=BytesIO(b"text"))
            with self.assertRaises(HTTPException):
                await read_text_upload(file)
            await file.close()

    async def test_rerank_non_object_json(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        for payload in ("[]", "null", "123"):
            client.post.return_value = httpx.Response(
                200, json={"response": payload},
                request=httpx.Request("POST", "http://test/api/generate"),
            )
            rs = await OllamaClient(Settings(), client).rerank_documents(
                "질문", [{"id": 1, "content": "내용"}]
            )
            self.assertEqual(rs, [])
