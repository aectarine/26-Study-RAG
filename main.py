import hashlib
import logging
import re
import time

import httpx
import psycopg
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database import get_connection
from embedding import create_embedding
from rag import generate_answer, rerank_documents

# 검색 결과 개수
SEARCH_LIMIT = 3
CANDIDATE_LIMIT = 5
# 거리 필터링 기준
DISTANCE_THRESHOLD = 0.5
DUPLICATE_DISTANCE_THRESHOLD = 0.1
NO_ANSWER_MESSAGE = "제공된 문서에서 관련 정보를 찾을 수 없습니다."


class EmbeddingRequest(BaseModel):
    text: str


class DocumentRequest(BaseModel):
    content: str


class SearchRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    id: int
    content: str
    distance: float
    source_document_id: int | None
    filename: str | None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]


app = FastAPI(
    title="Study-RAG",
    description="FastAPI + PostgreSQL + Ollama RAG 학습 프로젝트",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "운영 API",
            "description": "검증된 검색 전략을 사용하는 실제 서비스 API"
        },
        {
            "name": "테스트 API",
            "description": "검색 전략 비교와 성능 측정을 위한 실험 API"
        },
        {
            "name": "검색 API",
            "description": "답변 생성 없이 문서 검색만 수행하는 API"
        },
        {
            "name": "문서 관리",
            "description": "문서 업로드, 조회, 수정, 삭제 API"
        },
        {
            "name": "상태 확인",
            "description": "서버와 DB 상태를 확인하는 API"
        }
    ]
)

logger = logging.getLogger(__name__)


@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    request.state.timings = {}
    start_time = time.perf_counter()

    # 요청에 해당하는 API 실행
    response = await call_next(request)

    # API 실행 완료 후 경과 시간 계산
    elapsed_time = time.perf_counter() - start_time

    # 응답 헤더에 실행 시간 추가
    response.headers["X-Process-Time"] = f"{elapsed_time:.3f}"

    header_names = {
        "embedding-time": "X-Embedding-Time",
        "db-time": "X-DB-Time",
        "rerank-time": "X-Rerank-Time",
        "answer-time": "X-Answer-Time"
    }

    for name, value in request.state.timings.items():
        header_name = header_names.get(name)

        if header_name:
            response.headers[header_name] = f"{value:.3f}"

    return response


@app.exception_handler(httpx.ConnectError)
async def ollama_connection_error(request: Request, e: httpx.ConnectError):
    logger.error("Ollama 연결 실패: %s", e)
    return JSONResponse(
        status_code=503,
        content={
            "error": "ollama_unavailable",
            "message": "Ollama 서버에 연결할 수 없습니다."
        }
    )


@app.exception_handler(httpx.TimeoutException)
async def ollama_timeout_error(request: Request, e: httpx.TimeoutException):
    logger.error("Ollama 요청 시간 초과: %s", e)
    return JSONResponse(
        status_code=504,
        content={
            "error": "ollama_timeout",
            "message": "Ollama 답변 생성 시간이 초과되었습니다."
        }
    )


@app.exception_handler(psycopg.Error)
async def database_error(request: Request, e: psycopg.Error):
    logger.error("데이터베이스 오류: %s", e)
    return JSONResponse(
        status_code=503,
        content={
            "error": "database_unavailable",
            "message": "데이터베이스에 연결할 수 없습니다."
        }
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, e: Exception):
    logger.exception("처리되지 않은 서버 오류")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "서버 내부 오류가 발생했습니다."
        }
    )


def split_text(content: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    if overlap >= max_chars:
        raise ValueError("overlap은 max_chars보다 작아야 합니다.")

    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not content:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in content.split("\n\n")
        if paragraph.strip()
    ]

    heading_pattern = re.compile(r"^(?:#{1,6}\s+|\d+[.)]\s+).+")
    merged_paragraphs = paragraphs
    paragraphs = []
    index = 0

    while index < len(merged_paragraphs):
        paragraph = merged_paragraphs[index]
        is_heading = bool(heading_pattern.match(paragraph))

        if (
                is_heading
                and len(paragraph) <= 120
                and index + 1 < len(merged_paragraphs)
        ):
            paragraphs.append(
                f"{paragraph}\n{merged_paragraphs[index + 1]}"
            )
            index += 2
            continue

        paragraphs.append(paragraph)
        index += 1

    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        start = 0
        paragraph_length = len(paragraph)

        while start < paragraph_length:
            target_end = min(
                start + max_chars,
                paragraph_length
            )

            end = target_end

            if target_end < paragraph_length:
                minimum_boundary = start + max_chars // 2
                paragraph_part = paragraph[start:target_end]
                sentence_matches = list(
                    re.finditer(
                        r"[.!?](?=\s|$)",
                        paragraph_part
                    )
                )

                if sentence_matches:
                    sentence_end = start + sentence_matches[-1].end()

                    if sentence_end >= minimum_boundary:
                        end = sentence_end
                else:
                    space_boundary = paragraph.rfind(
                        " ",
                        minimum_boundary,
                        target_end
                    )

                    if space_boundary > start:
                        end = space_boundary

            chunk = paragraph[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= paragraph_length:
                break

            next_start = max(
                end - overlap,
                start + 1
            )

            while (
                    next_start < paragraph_length
                    and paragraph[next_start] != " "
            ):
                next_start += 1

            next_start = min(
                next_start + 1,
                paragraph_length
            )

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks


def create_content_hash(content: str) -> str:
    # Windows와 Linux의 줄바꿈 차이를 통일
    normalize_content = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalize_content.encode("utf-8")).hexdigest()


def remove_duplicate_documents(documents: list[dict]) -> list[dict]:
    unique_documents = []
    seen_contents = set()

    for document in documents:
        content = document["content"].strip()

        if content in seen_contents:
            continue

        seen_contents.add(content)
        unique_documents.append(document)

    return unique_documents


def remove_contained_documents(documents: list[dict]) -> list[dict]:
    """더 가까운 문서의 내용을 포함하는 긴 중복 청크를 제거합니다."""
    normalized_contents = [
        " ".join(document["content"].split())
        for document in documents
    ]
    unique_documents = []

    for index, document in enumerate(documents):
        current_content = normalized_contents[index]
        is_redundant = False

        for other_index, other_content in enumerate(normalized_contents):
            if index == other_index:
                continue

            if len(current_content) <= len(other_content):
                continue

            if normalized_contents[other_index] in current_content:
                if (
                        documents[other_index]["distance"]
                        <= document["distance"]
                ):
                    is_redundant = True
                    break

        if not is_redundant:
            unique_documents.append(document)

    return unique_documents


def row_to_document(row: tuple) -> dict:
    """DB 검색 결과를 문서 형식으로 변환합니다."""
    return {
        "id": row[0],
        "content": row[1],
        "distance": row[2],
        "source_document_id": row[3],
        "filename": row[4],
    }


async def retrieve_documents(
        question: str,
        strategy: str = "basic",
        timings: dict[str, float] | None = None,
) -> list[dict]:
    """
    검색 전략에 따라 관련 문서를 조회합니다.

    검색 전략:
    - basic:
        질문과 가까운 청크를 거리순으로 검색합니다.

    - filtered:
        코사인 거리가 DISTANCE_THRESHOLD 이하인 청크만 검색합니다.

    - deduplicated:
        상위 검색 결과를 가져온 뒤 Python에서 content 중복을 제거합니다.

    - deduplicated-db:
        PostgreSQL의 DISTINCT ON으로 DB에서 content 중복을 제거한 뒤
        거리순으로 정렬합니다.
    """

    allowed_strategies = {
        "basic",
        "filtered",
        "deduplicated",
        "deduplicated-db",
        "semantic-deduplicated"
    }

    if strategy not in allowed_strategies:
        raise ValueError(
            f"지원하지 않는 검색 전략입니다: {strategy}"
        )

    query_embedding = await create_embedding(question, timings)

    embedding_value = str(query_embedding)

    # --------------------------------------------------
    # DB 중복 제거 검색
    # --------------------------------------------------
    if strategy == "semantic-deduplicated":
        query = """
                WITH candidates AS (SELECT r.id,
                                           r.content,
                                           r.embedding,
                                           r.embedding <=> %s::vector AS query_distance,
                                           r.source_document_id,
                                           s.filename
                                    FROM rag_documents AS r
                                             LEFT JOIN source_documents AS s
                                                       ON r.source_document_id = s.id
                                    ORDER BY query_distance
                                    LIMIT %s),
                     duplicate_pairs AS (SELECT a.id             AS first_id,
                                                b.id             AS second_id,
                                                a.query_distance AS first_query_distance,
                                                b.query_distance AS second_query_distance
                                         FROM candidates AS a
                                                  CROSS JOIN candidates AS b
                                         WHERE a.id < b.id
                                           AND a.embedding <=> b.embedding <= %s),
                     removed_documents AS (SELECT CASE
                                                      WHEN first_query_distance <= second_query_distance
                                                          THEN second_id
                                                      ELSE first_id
                                                      END AS remove_id
                                           FROM duplicate_pairs)
                SELECT id,
                       content,
                       query_distance AS distance,
                       source_document_id,
                       filename
                FROM candidates
                WHERE id NOT IN (SELECT remove_id
                                 FROM removed_documents)
                ORDER BY query_distance
                LIMIT %s
                """

        query_parameters = (
            embedding_value,
            CANDIDATE_LIMIT,
            DUPLICATE_DISTANCE_THRESHOLD,
            SEARCH_LIMIT
        )
    elif strategy == "deduplicated-db":
        query = """
                SELECT d.id,
                       d.content,
                       d.distance,
                       d.source_document_id,
                       d.filename
                FROM (SELECT DISTINCT ON (r.content) r.id,
                                                     r.content,
                                                     r.embedding <=> %s::vector AS distance,
                                                     r.source_document_id,
                                                     s.filename
                      FROM rag_documents AS r
                               LEFT JOIN source_documents AS s
                                         ON r.source_document_id = s.id
                      ORDER BY r.content,
                               distance,
                               r.id) AS d
                ORDER BY d.distance,
                         d.id
                LIMIT %s \
                """

        query_parameters = (
            embedding_value,
            SEARCH_LIMIT,
        )

    # --------------------------------------------------
    # 거리 필터링 검색
    # --------------------------------------------------
    elif strategy == "filtered":
        query = """
                SELECT r.id,
                       r.content,
                       r.embedding <=> %s::vector AS distance,
                       r.source_document_id,
                       s.filename
                FROM rag_documents AS r
                         LEFT JOIN source_documents AS s
                                   ON r.source_document_id = s.id
                WHERE r.embedding <=> %s::vector <= %s
                ORDER BY distance
                LIMIT %s \
                """

        query_parameters = (
            embedding_value,
            embedding_value,
            DISTANCE_THRESHOLD,
            SEARCH_LIMIT,
        )

    # --------------------------------------------------
    # 기본 검색 및 Python 중복 제거용 검색
    # --------------------------------------------------
    else:
        query = """
                SELECT r.id,
                       r.content,
                       r.embedding <=> %s::vector AS distance,
                       r.source_document_id,
                       s.filename
                FROM rag_documents AS r
                         LEFT JOIN source_documents AS s
                                   ON r.source_document_id = s.id
                ORDER BY distance
                LIMIT %s \
                """

        query_parameters = (
            embedding_value,
            SEARCH_LIMIT,
        )

    db_started_at = time.perf_counter()

    # --------------------------------------------------
    # PostgreSQL 검색 실행
    # --------------------------------------------------
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                query_parameters,
            )
            rows = cursor.fetchall()

    if timings is not None:
        timings["db-time"] = time.perf_counter() - db_started_at

    # --------------------------------------------------
    # DB 결과를 API 응답 형식으로 변환
    # --------------------------------------------------
    documents = [
        row_to_document(row)
        for row in rows
    ]

    # --------------------------------------------------
    # Python 중복 제거
    # --------------------------------------------------
    if strategy == "deduplicated":
        documents = remove_duplicate_documents(documents)

    if strategy == "semantic-deduplicated":
        documents = remove_contained_documents(documents)

    return documents


@app.get("/", tags=["상태 확인"])
def root():
    return {"message": "Study-RAG Server Running"}


@app.get("/health", tags=["상태 확인"])
def health():
    return {"status": "ok"}


@app.get("/db-test", tags=["상태 확인"])
def db_test():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM rag_documents")
            count = cursor.fetchone()[0]

    return {
        "status": "connected",
        "document_count": count
    }


@app.post("/embedding-test", tags=["상태 확인"])
async def embedding_test(request: EmbeddingRequest):
    embedding = await create_embedding(request.text)
    return {
        "text": request.text,
        "dimension": len(embedding),
        "embedding_preview": embedding[:5]
    }


@app.post("/documents", tags=["문서 관리"])
async def create_document(request: DocumentRequest):
    # 1. 문장을 임베딩 벡터로 변환
    embedding = await create_embedding(request.content)

    # 2. PostgreSQL 저장
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_documents (content, embedding)
                VALUES (%s, %s::vector)
                RETURNING id
                """,
                (
                    request.content,
                    str(embedding)
                )
            )

            document_id = cursor.fetchone()[0]

    return {
        "document_id": document_id,
        "content": request.content,
        "embedding_dimension": len(embedding),
        "status": "saved"
    }


# 여기까지 과정은 RAG의 검색(Retrieval) 구현 완료
# 다음 과정은 검색된 문서를 Ollama의 qwen3:4b 모델에 전달하여 실제 AI 답변을 생성하도록 한다
@app.post("/search", tags=["검색 API"])
async def search_document(request: SearchRequest):
    documents = await retrieve_documents(request.question, "basic")
    return {
        "question": request.question,
        "documents": documents
    }


@app.post("/search/filtered", tags=["검색 API"])
async def search_document_filtered(request: SearchRequest):
    documents = await retrieve_documents(request.question, "filtered")
    return {
        "question": request.question,
        "documents": documents
    }


@app.post("/test/chat/basic", response_model=ChatResponse, tags=["테스트 API"])
async def chat_basic_test(http_request: Request, request: SearchRequest):
    timings = http_request.state.timings
    documents = await retrieve_documents(request.question, "basic", timings)

    if not documents:
        return {
            "question": request.question,
            "answer": NO_ANSWER_MESSAGE,
            "sources": []
        }

    answer = await generate_answer(
        question=request.question,
        documents=documents,
        timings=timings
    )

    response_sources = documents

    if answer.strip().startswith(NO_ANSWER_MESSAGE):
        response_sources = []

    return {
        "question": request.question,
        "answer": answer,
        "sources": response_sources
    }


async def create_chat_response(
        http_request: Request,
        request: SearchRequest,
        strategy: str
):
    timings = http_request.state.timings

    documents = await retrieve_documents(request.question, strategy, timings)

    if not documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 질문과 관련된 정보를 찾지 못했습니다.",
            "sources": []
        }

    answer = await generate_answer(request.question, documents, timings)

    response_sources = documents

    if answer.strip().startswith(NO_ANSWER_MESSAGE):
        response_sources = []

    # 6. AI 답변과 검색된 청크 반환
    return {
        "question": request.question,
        "answer": answer,
        "sources": response_sources
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["운영 API"]
)
async def chat_document(http_request: Request, request: SearchRequest):
    return await create_chat_response(
        http_request,
        request,
        "semantic-deduplicated"
    )


@app.post(
    "/test/chat/filtered",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_filtered(http_request: Request, request: SearchRequest):
    return await create_chat_response(
        http_request,
        request,
        "filtered"
    )


@app.post(
    "/test/chat/reranked",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_reranked(
        http_request: Request,
        request: SearchRequest
):
    timings = http_request.state.timings
    documents = await retrieve_documents(request.question, "filtered", timings)

    selected_documents = await rerank_documents(
        request.question,
        documents,
        timings
    )

    if not selected_documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 질문에 답할 수 있는 정보를 찾지 못했습니다.",
            "sources": []
        }

    answer = await generate_answer(
        request.question,
        selected_documents,
        timings
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": selected_documents
    }


@app.post(
    "/test/chat/deduplicated",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_deduplicated(
        http_request: Request,
        request: SearchRequest
):
    timings = http_request.state.timings
    documents = await retrieve_documents(request.question, "deduplicated", timings)

    if not documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 관련 정보를 찾지 못했습니다.",
            "sources": []
        }

    # 6. 중복 제거된 청크로 답변 생성
    answer = await generate_answer(request.question, documents, timings)

    return {
        "question": request.question,
        "answer": answer,
        "sources": documents
    }


@app.post(
    "/test/chat/deduplicated-db",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_deduplicated_db(
        http_request: Request,
        request: SearchRequest
):
    timings = http_request.state.timings
    documents = await retrieve_documents(
        request.question,
        "deduplicated-db",
        timings
    )

    if not documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 관련 정보를 찾지 못했습니다.",
            "sources": []
        }

    answer = await generate_answer(request.question, documents, timings)

    return {
        "question": request.question,
        "answer": answer,
        "sources": documents
    }


@app.post("/documents/upload", tags=["문서 관리"])
async def upload_document(file: UploadFile = File(...)):
    # 1. TXT 파일 확인 및 내용 읽기
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="TXT 파일만 업로드할 수 있습니다."
        )

    # 2. TXT 파일 읽기
    file_bytes = await file.read()

    try:
        content = file_bytes.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="UTF-8로 인코딩된 TXT 파일을 업로드해 주세요."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="파일 내용이 비어 있습니다."
        )

    # 3. 내용을 문단별로 분할
    chunks = split_text(content)

    # 4. 파일 본문 해시값 생성
    content_hash = create_content_hash(content)

    # 5. 같은 파일명으로 등록된 문서가 있는지 확인
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, content_hash
                FROM source_documents
                WHERE filename = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (file.filename,)
            )
            existing_document = cursor.fetchone()

    # 6. 기존 문서가 있다면 내용 비교
    if existing_document:
        existing_id, existing_hash = existing_document

        # 파일명과 내용이 모두 동일하면 중복 등록 생략
        if existing_hash == content_hash:
            return {
                "filename": file.filename,
                "source_document_id": existing_id,
                "status": "skipped",
                "message": "이미 등록된 동일한 문서입니다."
            }

        # 파일명은 같지만 내용이 다르면 자동으로 덮어쓰지 않음
        raise HTTPException(
            status_code=409,
            detail=(
                "같은 파일명으로 등록된 문서가 있지만 내용이 다릅니다. "
                "기존 문서 교체 또는 새 문서 등록을 선택해야 합니다."
            )
        )

    # 7. 각 청크의 임베딩 생성
    embeddings = []

    for chunk in chunks:
        embedding = await create_embedding(chunk)
        embeddings.append(embedding)

    # 8. 원본 파일 정보와 청크를 DB에 저장
    saved_documents = []

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 원본 파일 정보 저장
            cursor.execute(
                """
                INSERT INTO source_documents (filename, content_hash)
                VALUES (%s, %s)
                RETURNING id
                """,
                (file.filename, content_hash)
            )

            source_document_id = cursor.fetchone()[0]

            # 각 청크와 임베딩 저장
            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO rag_documents (source_document_id, content, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id
                    """,
                    (
                        source_document_id,
                        chunk,
                        embedding
                    )
                )

                document_id = cursor.fetchone()[0]

                saved_documents.append({
                    "document_id": document_id,
                    "content": chunk
                })

    # 9. 저장 결과 반환
    return {
        "filename": file.filename,
        "source_document_id": source_document_id,
        "chunk_count": len(saved_documents),
        "documents": saved_documents,
        "status": "saved"
    }


@app.put("/documents/{source_document_id}", tags=["문서 관리"])
async def replace_document(
        source_document_id: int,
        force: bool = False,
        file: UploadFile = File(...)
):
    # 1. TXT 파일 확인
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="TXT 파일만 업로드 가능합니다."
        )

    # 2. 파일 내용 읽기
    file_bytes = await file.read()

    try:
        content = file_bytes.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="UTF-8로 인코딩된 TXT 파일을 업로드 해주세요."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="파일 내용이 비어 있습니다."
        )

    # 3. 기존 문서 조회
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT filename, content_hash
                FROM source_documents
                WHERE id = %s
                """,
                (source_document_id,)
            )

            existing_document = cursor.fetchone()

    if existing_document is None:
        raise HTTPException(
            status_code=404,
            detail="해당 문서를 찾을 수 없습니다."
        )

    # 4. 본문 해시 비교
    new_hash = create_content_hash(content)
    old_hash = existing_document[1]

    if old_hash == new_hash and not force:
        return {
            "source_document_id": source_document_id,
            "status": "skipped",
            "message": (
                "기존 문서와 내용이 동일합니다. "
                "청크 정책을 다시 적용하려면 force=true를 사용하세요."
            )
        }

    # 5. 새 문서 분할 및 임베딩 생성
    chunks = split_text(content)

    embeddings = []

    for chunk in chunks:
        embedding = await create_embedding(chunk)
        embeddings.append(embedding)

    # 6. 기존 청크 교체 및 원본 정보 갱신
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 교체 대상 문서를 잠금
            cursor.execute(
                """
                SELECT id
                FROM rag_documents
                WHERE source_document_id = %s
                    FOR UPDATE
                """,
                (source_document_id,)
            )

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="해당 문서를 찾을 수 없습니다."
                )

            # 기존 청크 삭제
            cursor.execute(
                """
                DELETE
                FROM rag_documents
                WHERE source_document_id = %s
                """,
                (source_document_id,)
            )

            # 새 청크 저장
            saved_documents = []

            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO rag_documents (source_document_id, content, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id
                    """,
                    (source_document_id, chunk, str(embedding))
                )

                document_id = cursor.fetchone()[0]

                saved_documents.append({
                    "document_id": document_id,
                    "content": chunk
                })

            # 원본 문서 정보 갱신
            cursor.execute(
                """
                UPDATE source_documents
                SET filename     = %s,
                    content_hash = %s
                WHERE id = %s
                """,
                (file.filename, new_hash, source_document_id)
            )

        # 7. 교체 결과 반환
        return {
            "filename": file.filename,
            "source_document_id": source_document_id,
            "chunk_count": len(saved_documents),
            "documents": saved_documents,
            "status": "replaced"
        }


@app.delete("/documents/{source_document_id}", tags=["문서 관리"])
def delete_document(source_document_id: int):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 1. 삭제할 원본 문서 확인
            cursor.execute(
                """
                SELECT id, filename
                FROM source_documents
                WHERE id = %s
                    FOR UPDATE
                """,
                (source_document_id,)
            )

            document = cursor.fetchone()

            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail="해당 문서를 찾을 수 없습니다."
                )

            # 2. 연결된 청크 삭제
            cursor.execute(
                """
                DELETE
                FROM rag_documents
                WHERE source_document_id = %s
                """,
                (source_document_id,)
            )

            deleted_chunk_count = cursor.rowcount

            # 3. 원본 문서 삭제
            cursor.execute(
                """
                DELETE
                FROM source_documents
                WHERE id = %s
                """,
                (source_document_id,)
            )

    return {
        "source_document_id": source_document_id,
        "filename": document[1],
        "deleted_chunk_count": deleted_chunk_count,
        "status": "deleted"
    }


@app.get("/documents", tags=["문서 관리"])
def get_documents():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT s.id, s.filename, s.inserted, COUNT(r.id) as chunk_count
                FROM source_documents as s
                         LEFT JOIN rag_documents as r
                                   ON r.source_document_id = s.id
                GROUP BY s.id
                ORDER BY s.id DESC
                """
            )

            rows = cursor.fetchall()

    return {
        "total": len(rows),
        "documents": [
            {
                "source_document_id": row[0],
                "filename": row[1],
                "inserted": row[2],
                "chunk_count": row[3]
            }
            for row in rows
        ]
    }


@app.get("/documents/{source_document_id}", tags=["문서 관리"])
def get_document(source_document_id: int):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, filename, content_hash, inserted
                FROM source_documents
                WHERE id = %s
                """,
                (source_document_id,)
            )

            document = cursor.fetchone()

            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail="해당 문서를 찾을 수 없습니다."
                )

            # 해당 문서에 연결된 청크 조회
            cursor.execute(
                """
                SELECT id, content
                FROM rag_documents
                WHERE source_document_id = %s
                ORDER BY id
                """,
                (source_document_id,)
            )

            chunks = cursor.fetchall()

    return {
        "source_document_id": document[0],
        "filename": document[1],
        "content_hash": document[2],
        "inserted": document[3],
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk[0],
                "content": chunk[1]
            }
            for chunk in chunks
        ]
    }


@app.post(
    "/test/chat/semantic-deduplicated",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_deduplicated_semantic(
        http_request: Request,
        request: SearchRequest
):
    timings = http_request.state.timings
    documents = await retrieve_documents(
        request.question,
        "semantic-deduplicated",
        timings
    )

    if not documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 관련 정보를 찾지 못했습니다.",
            "sources": []
        }

    answer = await generate_answer(
        request.question,
        documents,
        timings
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": documents
    }


@app.post(
    "/test/chat/semantic-reranked",
    response_model=ChatResponse,
    tags=["테스트 API"]
)
async def chat_document_deduplicated_semantic_reranked(
        http_request: Request,
        request: SearchRequest
):
    timings = http_request.state.timings
    documents = await retrieve_documents(
        request.question,
        "semantic-deduplicated",
        timings
    )

    selected_documents = await rerank_documents(
        request.question,
        documents,
        timings
    )

    if not selected_documents:
        return {
            "question": request.question,
            "answer": "등록된 문서에서 질문에 답할 수 있는 정보를 찾지 못했습니다.",
            "sources": []
        }

    answer = await generate_answer(
        request.question,
        selected_documents,
        timings
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": selected_documents
    }
