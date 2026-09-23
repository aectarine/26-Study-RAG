# Study-RAG

FastAPI, PostgreSQL/pgvector, Ollama를 이용해 문서를 저장하고 문서 청크의 임베딩을 기반으로 검색·답변하는 학습용 RAG 서비스입니다. 애플리케이션 코드는 `Controller(router) → Service → Repository` 계층으로 분리하고, SQLAlchemy 세션과 서비스 트랜잭션을 사용합니다.

현재 프로젝트는 TXT 문서 업로드부터 임베딩 저장, 유사도 검색, 검색 결과별 답변 생성, 응답 시간 측정, 운영·테스트 API 분리까지 구현되어 있습니다. 외부 API는 모두 `/api` 아래에 있으며, 운영 API `/api/chat`은 DB 벡터 거리 기반 의미 중복 제거 전략을 사용합니다.

## 목차

- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 현재 실행 환경](#2-현재-실행-환경)
- [3. 전체 처리 흐름](#3-전체-처리-흐름)
- [4. 프로젝트 구조](#4-프로젝트-구조)
- [5. 현재 구현된 기능](#5-현재-구현된-기능)
  - [문서 관리](#문서-관리)
  - [검색과 답변](#검색과-답변)
  - [응답 시간 측정](#응답-시간-측정)
- [6. API 목록](#6-api-목록)
- [7. 검색 전략](#7-검색-전략)
- [8. 데이터 구조](#8-데이터-구조)
- [9. 실행 방법](#9-실행-방법)
- [10. 테스트 방법](#10-테스트-방법)
- [11. 실험 결과와 한계](#11-실험-결과와-한계)
- [12. 현재 진행 상황](#12-현재-진행-상황)
- [13. 남은 작업 목록](#13-남은-작업-목록)
- [14. 다음 진행 순서](#14-다음-진행-순서)
- [15. 확인 필요 항목](#15-확인-필요-항목)

## 1. 프로젝트 개요

### 목표

RAG의 핵심 과정을 직접 구현하고 검색 방식에 따른 품질과 응답 시간의 차이를 비교합니다.

### 주요 기술

| 구분 | 기술 |
|---|---|
| API 서버 | FastAPI + fastapi-utils CBV |
| 데이터베이스 | PostgreSQL |
| DB 접근 계층 | SQLAlchemy 2.x AsyncSession + asyncpg |
| 벡터 검색 | pgvector, 코사인 거리 연산자 `<=>` |
| 임베딩·답변 생성 | Ollama |
| 패키지 관리 | uv |
| 의존성 주입 | dependency-injector |
| 실행 환경 | Windows + Docker Desktop |

## 2. 현재 실행 환경

| 구성 요소 | 현재 상태 | 확인 근거 |
|---|---|---|
| FastAPI | Windows에서 실행 | `main.py`, `router/` |
| PostgreSQL/pgvector | Docker Desktop에서 실행 | 사용자 확인 |
| Ollama | Windows에 직접 설치 | 사용자 확인 |
| Ollama 주소 | `http://localhost:11434` | `core/cfg/settings.py` |
| 임베딩 모델 | `embeddinggemma` | `core/embed/embedding.py`, Ollama 설치 상태 별도 확인 필요 |
| 답변 생성 모델 | `qwen2.5:3b` | `.env.dev` → 중앙 설정 객체 |
| 추론 답변 모델 | `study-rag-llm:latest` | 별도 실험용 |
| 리랭킹 모델 | `qwen3:4b` | `core/llm/ollama.py` |
| Python | 3.14 이상 | `pyproject.toml` |

DB 접속 정보는 다음 환경변수를 사용합니다.

```text
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
```

비밀번호와 실제 접속 값은 README에 기록하지 않습니다.

### Ollama 모델 구성

현재 Windows Ollama에 확인된 모델은 다음과 같습니다.

| 모델 | 역할 | 상태 |
|---|---|---|
| `study-rag-llm:latest` | 추론 기능을 포함한 최종 답변 생성 후보 | 설치 확인 |
| `qwen3:4b` | 리랭킹 및 `study-rag-llm`의 기반 모델 | 설치 확인 |
| `qwen2.5:3b` | 현재 코드에서 사용하는 답변 생성 모델 | 설치 확인 |
| `embeddinggemma` | 질문·문서 임베딩 생성 | 코드 설정 확인, 설치 상태 확인 필요 |

`Modelfile`은 `qwen3:4b`를 기반으로 한국어 문서 기반 답변 시스템 프롬프트와 `temperature 0.3`을 설정합니다. 현재 운영 답변 생성은 `qwen2.5:3b`를 사용하며, `study-rag-llm:latest`는 추론 모델 비교용으로 보류되어 있습니다.

## 3. 전체 처리 흐름

```text
TXT 업로드
    ↓
문단·명시적 제목 결합 및 청크 분할
    ↓
Ollama 임베딩 생성
    ↓
PostgreSQL + pgvector 저장
    ↓
사용자 질문 임베딩 생성
    ↓
코사인 거리 기반 청크 검색
    ↓
필터링·중복 제거·리랭킹
    ↓
Ollama 답변 생성
    ↓
답변과 출처 반환
```

## 4. 프로젝트 구조

| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 앱 생성, 미들웨어·예외 처리·라우터 등록 |
| `core/cfg/settings.py` | 환경변수 기반 싱글톤 설정 객체 |
| `core/db/database.py` | SQLAlchemy 비동기 엔진·세션 팩토리와 DB 수명주기 |
| `core/dep/container.py` | Singleton·Factory 정의, 요청별 세션과 객체 조립·정리 통합 관리 |
| `core/util/route.py` | CBV 그룹 루트를 기존 슬래시 없는 API 경로로 등록 |
| `core/model/entity.py` | SQLAlchemy ORM 모델과 pgvector 타입 정의 |
| `core/schema/request.py` | Pydantic 요청 스키마(`EmbeddingRequest`, `DocumentRequest`, `SearchRequest`) |
| `core/schema/response.py` | Pydantic 응답 스키마(`ChatResponse`, `SourceDocumentResponse`) |
| `core/handler/exception.py` | 공통 예외 응답 등록 |
| `core/middleware/timing.py` | 처리 시간 헤더와 요청 로그 미들웨어 |
| `core/util/document.py` | TXT 검증·해시·청크 분할 유틸리티 |
| `core/util/retrieval.py` | DI 컨테이너를 사용하는 검색 호출 유틸리티 |
| `repo/document_repo.py` | 문서·청크 CRUD와 벡터 검색 Repository |
| `service/document_service.py` | 문서 생성·업로드·조회·교체·삭제 업무와 트랜잭션 경계 |
| `service/chunk_service.py` | 상위 서비스 트랜잭션에 참여하는 청크 저장·삭제 |
| `service/rag_service.py` | 임베딩 검색·필터·중복 제거 업무 |
| `service/chat_service.py` | 검색 결과 기반 답변 생성 업무 |
| `router/status_router.py` | 상태 확인·임베딩 테스트 Controller |
| `router/search_router.py` | 검색 Controller |
| `router/chat_router.py` | 운영·테스트 채팅 Controller |
| `router/document_router.py` | 문서 관리 Controller |
| `core/embed/embedding.py` | Ollama 임베딩 API 호출(`create_embedding`) |
| `core/llm/ollama.py` | Ollama 답변 생성 및 리랭킹 |
| `test/test_search_benchmark.py` | 검색 API 5회 반복 벤치마크 |
| `test/test_quality.py` | 운영 API 답변 내용과 기대 출처 검증, 6개 질문 유형 |
| `test/test_document_api.py` | 문서 업로드·조회·교체·삭제 회귀 테스트 |
| `test/test_error_api.py` | 잘못된 요청과 존재하지 않는 문서의 오류 응답 검증 |
| `test/test_fault_api.py` | 모킹 기반 DB·Ollama 장애 응답 검증 |
| `test/test_architecture.py` | 격리된 DB에서 요청 DI·공동 롤백·취소·자원 정리 검증 |
| `test/test_unit.py` | 청크 누락 방지·업로드 읽기 제한·DB URL·리랭킹 경계값 검증 |
| `test/test_concurrency.py` | `/chat` 동시 요청과 응답 헤더 검증 |
| `router/` | 운영·실험 API Controller 분리 |
| `service/` | 검색·답변·문서 업무 로직과 트랜잭션 처리 |
| `repo/` | SQLAlchemy Repository와 영속성 처리 |
| `core/` | 설정·DB·DI 공통 기반 |
| `Modelfile` | Ollama 모델 설정 예시 |
| `test/test_main.http` | HTTP 요청 테스트 예시 |
| `pyproject.toml` | Python 의존성과 프로젝트 설정 |
| `uv.lock` | 의존성 잠금 파일 |
| `.env.dev` | Git에 포함되는 개발 환경변수 |
| `.env` | 로컬 전용 환경변수 |

각 애플리케이션 폴더는 Python namespace package로 사용하므로 재수출만 하는 `__init__.py`를 두지 않습니다. 생성 규칙과 요청별 조립은 `core/dep/container.py` 하나에서 관리합니다. 상단의 Container는 Singleton·Factory를 정의하고 하단의 의존성 함수는 요청별 세션 생성·정리와 객체 재사용을 담당합니다. 라우터는 `@cbv` 클래스 속성에서 서비스를 한 번 선언하여 각 메서드에서 `self.service`로 사용합니다.

최종 계층 의존 방향은 다음과 같습니다.

```text
main.py
  └─ router/*_router.py       API 경로·요청 검증·응답 변환
       └─ service/*_service.py 업무 규칙·트랜잭션 경계
            └─ repo/*_repo.py  SQLAlchemy 조회·저장·삭제
                 └─ core/model/entity.py

core/dep/container.py          Singleton·Factory 정의와 요청별 객체 조립·세션 종료
core/embed/                    Ollama 임베딩 클라이언트
core/llm/                      Ollama 답변·리랭킹 클라이언트
core/util/                     문서·검색 공통 유틸리티
```

라우터는 Repository를 직접 호출하지 않습니다. Service와 Repository는 요청마다 생성하고 같은 요청에서는 FastAPI 의존성 캐시로 재사용합니다. 설정·DB 엔진·세션 팩토리·HTTP 클라이언트만 프로세스별 Singleton입니다. Repository는 요청별 `AsyncSession`을 생성자로 받으므로 업무 메서드마다 세션이나 UoW를 전달하지 않습니다.

### 객체 수명과 트랜잭션 규칙

| 객체 | 수명 | 공유 범위 |
|---|---|---|
| 설정·DB 엔진·연결 풀·HTTP 클라이언트 | Singleton | 서버 프로세스 |
| Service·Repository·AsyncSession | 요청별 | 동일 요청의 의존성 그래프 |
| DB 연결 | 풀에서 대여·반납 | 짧은 DB 트랜잭션 |

최상위 업무 서비스가 `async with self._session.begin()`을 시작하고 하위 서비스는 같은 세션으로 참여합니다. 예를 들어 `DocumentService`에서 문서를 저장한 뒤 `ChunkService`가 청크를 저장하면 둘 중 하나가 실패할 때 모두 롤백됩니다. 하위 서비스는 `begin()`, `commit()`을 호출하지 않습니다. 자체 트랜잭션을 여는 진입 메서드끼리 중첩 호출하지 말고 참여용 메서드로 분리합니다. 자동 전파 데코레이터는 사용하지 않습니다.

같은 세션을 사용하는 업무는 순차 실행하며 `asyncio.gather()`로 병렬 DB 작업을 실행하지 않습니다. 임베딩과 답변 생성은 DB 트랜잭션 밖에서 수행합니다. 문서 교체·삭제는 쓰기 시 원본 행을 잠그며, 요청 종료 시 세션을 닫고 서버 종료 시 HTTP 클라이언트와 DB 엔진을 정리합니다.

### 저사양 환경 연결 제한

| 환경변수 | 기본값 | 의미 |
|---|---:|---|
| `DB_POOL_SIZE` | 5 | 프로세스별 DB 풀 크기 |
| `DB_MAX_OVERFLOW` | 0 | 풀 외 추가 연결 허용 수 |
| `DB_POOL_TIMEOUT` | 10 | DB 연결 대기 제한(초) |
| `HTTP_MAX_CONNECTIONS` | 2 | 공유 Ollama HTTP 풀 최대 연결 수 |
| `HTTP_POOL_TIMEOUT` | 5 | HTTP 연결 대기 제한(초), 초과 시 503 |

풀 제한은 전체 대기 요청 수를 제한하지 않습니다. 운영에서는 서버 동시성 제한도 함께 설정합니다. 다음은 단일 프로세스 시작 예시이며 실제 용량에 맞게 조정해야 합니다.

```powershell
uv run uvicorn main:app --env-file .env.dev --limit-concurrency 20
```

워커 수를 늘리면 풀과 Singleton도 워커마다 생성됩니다. DB 최대 연결 수는 워커 수 × (`DB_POOL_SIZE` + `DB_MAX_OVERFLOW`)와 다른 애플리케이션 연결까지 합산해 산정합니다.

DB 접근은 `core/db/database.py`의 SQLAlchemy `AsyncSession`과 `asyncpg` 드라이버를 사용하며, 연결 URL은 `postgresql+asyncpg://` 형식입니다. 쓰기 작업은 서비스에서 `async with session.begin()`으로 트랜잭션을 처리합니다. 라우터는 SQL을 직접 실행하지 않습니다.

## 5. 현재 구현된 기능

### 문서 관리

- TXT 파일 업로드 및 UTF-8 검증
- 파일명 경로 이탈 차단 및 최대 5MB 업로드 제한
- 빈 줄 기준 문단 분할
- `# 제목`, `1. 제목`처럼 명시된 제목을 다음 본문과 결합
- 700자 초과 문단은 문장·공백 경계와 100자 overlap을 사용해 추가 분할
- 각 청크의 임베딩 생성 및 저장
- 파일명과 본문 해시를 이용한 동일 문서 중복 확인
- 문서 목록·상세 조회
- 문서 교체 및 삭제

### 검색과 답변

- 기본 유사도 검색
- 거리 임계값 필터링
- Ollama 리랭킹
- Python 기반 내용 중복 제거
- DB 기반 `DISTINCT ON` 중복 제거
- DB 벡터 거리와 문자열 포함 관계 기반 의미 중복 제거
- 검색 결과가 없을 때 답변 생성 생략
- 답변과 검색 출처(`sources`) 반환

### 응답 시간 측정

FastAPI 미들웨어가 모든 응답에 다음 헤더를 추가합니다.

```text
X-Process-Time: 0.123
```

API 전체 처리 시간과 임베딩, DB 검색, 리랭킹, 답변 생성 단계별 시간을 응답 헤더로 측정합니다.
또한 서버 로그에 요청 메서드, 경로, 상태 코드, 전체 처리 시간을 기록합니다. 요청 본문이나 환경변수 값은 로그에 기록하지 않습니다.

## 6. API 목록

API의 상위 경로와 기능별 기준 경로는 `main.py`에서 고정합니다. 운영 API는 `/api/chat`, `/api/search`, `/api/documents`를 사용하고, 테스트 기능은 `/api/test` 아래에 둡니다. 라우터 파일은 각 기준 경로 아래의 상세 경로만 선언합니다.

### 상태·기본 기능

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/` | 서버 실행 확인 |
| GET | `/api/health` | 상태 확인 |
| GET | `/api/test/db-test` | DB 연결 및 청크 수 확인 |
| POST | `/api/test/embedding-test` | 임베딩 생성 테스트 |

### 문서 관리

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/documents/upload` | TXT 업로드 및 청크·임베딩 저장 |
| GET | `/api/documents` | 문서 목록 조회 |
| GET | `/api/documents/{source_document_id}` | 문서 상세 조회 |
| PUT | `/api/documents/{source_document_id}` | 문서 교체 및 재임베딩, `force=true`로 동일 내용도 재처리 |
| DELETE | `/api/documents/{source_document_id}` | 문서와 연결 청크 삭제 |

### 검색·챗 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/chat` | DB 벡터 거리 기반 의미 중복 제거 후 답변 생성하는 운영 API |
| POST | `/api/search` | 거리순 상위 3개 검색 |

### 테스트 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/test/chat/basic` | 기본 검색 실험 |
| POST | `/api/test/chat/filtered` | 거리 필터링 실험 |
| POST | `/api/test/chat/reranked` | 거리 필터링 후 리랭킹 실험 |
| POST | `/api/test/chat/deduplicated` | Python 중복 제거 실험 |
| POST | `/api/test/chat/deduplicated-db` | DB 중복 제거 실험 |
| POST | `/api/test/chat/semantic-deduplicated` | 의미상 중복 제거 실험 |
| POST | `/api/test/chat/semantic-reranked` | 의미상 중복 제거 후 리랭킹 실험 |
| GET | `/api/test/db-test` | DB 연결 테스트 |
| POST | `/api/test/embedding-test` | 임베딩 생성 테스트 |
| POST | `/api/test/search/filtered` | 거리 필터링 검색 실험 |

초기 실험용 오타 엔드포인트는 제거했습니다. 신규 기능에서는 `/api/documents/upload`를 사용합니다.

### 현재 라우터 구조

Swagger 문서에서는 태그 순서를 `운영 API → 테스트 API → 검색 API → 문서 관리 → 상태 확인`으로 고정해 운영 API와 실험 API가 섞이지 않도록 구성했습니다.

| 구분 | 목표 경로 | 용도 |
|---|---|---|
| 운영 API | `/api/chat` | 최종 검증된 검색 전략만 제공 |
| 실험 API | `/api/test/chat/basic` | 기본 검색 비교 |
| 실험 API | `/api/test/chat/filtered` | 거리 필터링 비교 |
| 실험 API | `/api/test/chat/deduplicated` | Python 중복 제거 비교 |
| 실험 API | `/api/test/chat/deduplicated-db` | DB 중복 제거 비교 |
| 실험 API | `/api/test/chat/semantic-deduplicated` | 의미상 중복 제거 비교 |
| 실험 API | `/api/test/chat/reranked` | 거리 필터링 후 리랭킹 비교 |
| 실험 API | `/api/test/chat/semantic-reranked` | 의미상 중복 제거·리랭킹 비교 |

운영 API `/api/chat`의 검색 전략은 내부 서비스 함수에서 관리하고, 외부 경로는 안정적으로 유지합니다. 실험 API는 `/api/test` 아래에서 검색 방식 비교와 벤치마크를 위해 유지합니다.

대표 요청:

```json
{
  "question": "ESS 배터리의 적정 운영 온도와 보관 온도를 비교해줘."
}
```

대표 응답:

```json
{
  "question": "질문",
  "answer": "업로드 문서 기반 답변",
  "sources": [
    {
      "id": 7,
      "content": "검색된 청크",
      "distance": 0.123,
      "source_document_id": 1,
      "filename": "업로드 문서.txt"
    }
  ]
}
```

## 7. 검색 전략

| 전략 | 처리 방식 | 특징 |
|---|---|---|
| 기본 검색 | 거리순 상위 3개 선택 | 가장 단순하고 빠름 |
| 거리 필터링 | 거리 `<= 0.5`만 사용 | 관련 없는 결과를 줄일 수 있음 |
| 리랭킹 | 검색 결과를 Ollama가 재선별 | 품질 개선 가능, 추가 지연 발생 |
| Python 중복 제거 | 상위 결과를 문자열 기준 제거 | 구현이 단순하지만 결과 수 감소 가능 |
| DB 중복 제거 | `DISTINCT ON` 후 거리순 정렬 | 중복을 줄이고 다음 결과를 확보할 수 있음 |
| 의미상 중복 제거 | 임베딩 거리와 문자열 포함 관계로 중복 후보 제거 | 추가 Ollama 호출 없이 중복 출처 감소 |
| 의미상 중복 제거·리랭킹 | 의미상 중복 제거 후 Ollama 재선별 | 품질 비교용, 가장 느림 |

거리 필터링만으로 의미상 유사하지만 질문에 직접 필요하지 않은 청크를 모두 제거할 수는 없습니다.

## 8. 데이터 구조

코드에서 사용하는 논리적 테이블은 다음과 같습니다.

### `source_documents`

| 필드 | 용도 | 확인 상태 |
|---|---|---|
| `id` | 원본 문서 식별자 | 코드에서 사용 |
| `filename` | 업로드 파일명 | 코드에서 사용 |
| `content_hash` | 본문 중복 확인용 해시 | 코드에서 사용 |
| `inserted` | 등록 시각 | 조회 SQL에서 사용 |

### `rag_documents`

| 필드 | 용도 | 확인 상태 |
|---|---|---|
| `id` | 청크 식별자 | 코드에서 사용 |
| `source_document_id` | 원본 문서 참조 | 코드에서 사용 |
| `content` | 청크 텍스트 | 코드에서 사용 |
| `embedding` | 벡터 임베딩 | pgvector `vector` 사용 |

현재 확인된 DB 설정은 `embedding vector(768)`이며, `rag_documents.embedding`에는 HNSW와 IVFFlat 인덱스가 모두 생성되어 있습니다. PostgreSQL이 실행 계획에 따라 적절한 인덱스 또는 순차 스캔을 선택합니다. 데이터가 적을 때 순차 스캔이 선택되는 것은 정상입니다.

## 9. 실행 방법

### 사전 조건

1. Docker Desktop에서 PostgreSQL/pgvector 컨테이너 실행
2. `.env.dev`에 DB 접속 정보 설정
3. Windows에서 Ollama 실행
4. 필요한 Ollama 모델 준비

```powershell
ollama list
```

필요 모델:

```text
embeddinggemma
qwen2.5:3b
qwen3:4b
study-rag-llm:latest
```

### FastAPI 실행

```powershell
uv run uvicorn main:app --reload --env-file .env.dev
```

`--env-file`은 Uvicorn이 지정한 파일의 환경변수를 서버 실행 전에 주입하는 옵션입니다. 애플리케이션 코드는 특정 파일명을 직접 지정하지 않고 `os.getenv()`로 주입된 값을 읽습니다.

현재 로컬 검증 포트는 `8000`입니다.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## 10. 테스트 방법

1. `GET /api/health`로 서버 상태 확인
2. `GET /api/test/db-test`로 DB 연결 확인
3. `POST /api/documents/upload`로 TXT 업로드
4. 문서 목록과 상세 조회로 저장 확인
5. `POST /api/search`로 기본 검색 확인
6. 운영 API `/api/chat`과 일곱 가지 `/api/test/chat/*` 엔드포인트에 같은 질문 전송
7. `answer`, `sources`, `distance`, `filename` 비교
8. `X-Process-Time`, `X-Embedding-Time`, `X-DB-Time`, `X-Rerank-Time`, `X-Answer-Time` 값 비교
9. `test/test_document_api.py`로 문서 관리 API 회귀 테스트 실행
10. `test/test_error_api.py`로 `400`, `404`, `413` 오류 응답 검증
11. `test/test_fault_api.py`로 DB·Ollama 장애 응답 검증
12. `test/test_concurrency.py`로 동시 요청 처리 검증

핵심 테스트 명령은 다음과 같습니다.

```powershell
uv run python test/test_quality.py
uv run python test/test_document_api.py
uv run python test/test_error_api.py
uv run python test/test_fault_api.py
uv run python test/test_concurrency.py
uv run python test/test_search_benchmark.py
```

테스트 질문:

```json
{
  "question": "ESS 배터리의 적정 운영 온도와 보관 온도를 비교해줘."
}
```

문서에 없는 질문:

```json
{
  "question": "ESS 배터리의 제조사는 어디야?"
}
```

마지막 질문은 문서에 없는 정보를 모델이 만들어내지 않는지 확인하기 위한 질문입니다.

## 11. 실험 결과와 한계

이 절의 과거 벤치마크 표는 `/api` 공통 prefix를 적용하기 전 실행 결과라 경로 표기에서 prefix가 생략되어 있습니다. 현재 실행 경로는 모두 `/api/...`입니다.

| 관찰 내용 | 의미 |
|---|---|
| 동일하거나 의미가 유사한 청크가 상위 3개를 차지함 | 검색 결과가 중복으로 채워질 수 있음 |
| Python 중복 제거 후 결과 수가 감소함 | 다음 순위 결과를 자동으로 채우지 못함 |
| 의미상 중복 제거 후 출처 수가 감소함 | 답변에 필요한 핵심 출처만 전달할 수 있음 |
| 거리 필터링만으로 불필요한 청크가 남음 | 거리와 질문 관련성은 동일하지 않음 |
| 리랭킹 결과가 더 적절한 경우가 있음 | 추가 모델 호출 비용이 발생함 |
| 실행마다 응답 시간이 달라짐 | 모델 상태·캐시·실행 시점의 영향을 받음 |

### 5회 반복 벤치마크 결과

질문은 `ESS 배터리의 정기 점검 주기와 최대 충전량을 알려줘.`를 사용했습니다. 각 API를 5회씩 실행했으며, 모든 API가 200 응답을 반환했습니다.

#### 기존 모델 기준선

기존 `qwen2.5:3b` 답변 생성 모델을 사용한 기준선입니다.

| API | 평균 | 중앙값 | 표준편차 | 최소 | 최대 | source 수 | 중복 수 | 결과 청크 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/chat` | 1.964초 | 1.536초 | 0.840초 | 1.522초 | 3.642초 | 3 | 0 | 14, 9, 15 |
| `/chat/filtered` | 1.500초 | 1.505초 | 0.014초 | 1.482초 | 1.519초 | 3 | 0 | 14, 9, 15 |
| `/chat/reranked` | 15.981초 | 15.305초 | 2.043초 | 13.450초 | 18.800초 | 2 | 0 | 14, 15 |
| `/chat/deduplicated` | 1.880초 | 1.519초 | 0.735초 | 1.502초 | 3.350초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-db` | 1.531초 | 1.528초 | 0.021초 | 1.509초 | 1.565초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-semantic-reranked` | 20.959초 | 19.885초 | 2.997초 | 18.192초 | 26.727초 | 2 | 0 | 14, 15 |

#### `study-rag-llm:latest` 기준 결과

2026-09-22에 `study-rag-llm:latest`를 답변 생성 모델로 사용해 다시 측정했습니다. 모든 API가 200 응답을 반환했고, 모든 답변이 질문의 핵심 내용인 점검 주기 30일과 최대 충전량 85% 이하를 정확히 포함했습니다.

| API | 평균 | 중앙값 | 표준편차 | 최소 | 최대 | source 수 | 중복 수 | 결과 청크 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/chat` | 26.759초 | 25.895초 | 1.968초 | 24.534초 | 29.717초 | 3 | 0 | 14, 9, 15 |
| `/chat/filtered` | 30.449초 | 29.209초 | 12.180초 | 19.194초 | 53.142초 | 3 | 0 | 14, 9, 15 |
| `/chat/reranked` | 21.876초 | 24.035초 | 10.226초 | 2.666초 | 33.399초 | 2 | 0 | 14, 15 |
| `/chat/deduplicated` | 34.902초 | 33.544초 | 9.369초 | 21.079초 | 47.958초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-db` | 28.345초 | 27.603초 | 3.837초 | 22.658초 | 34.100초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-semantic-reranked` | 35.220초 | 23.906초 | 22.274초 | 22.477초 | 79.620초 | 2 | 0 | 14, 15 |

이번 결과의 핵심은 답변 품질은 유지됐지만, 추론 기능이 포함된 새 모델로 교체하면서 전체 응답 시간이 크게 증가했다는 점입니다. `/chat` 기준 평균은 기존 1.964초에서 26.759초로 증가했습니다. 또한 리랭킹이 포함된 방식은 의미상 중복 청크 9를 제외했지만 추가 모델 호출과 추론 시간 때문에 응답 시간 편차가 커졌습니다. 특히 `/chat/deduplicated-semantic-reranked`는 최대 79.620초까지 증가했으므로 운영 API 후보로 바로 선택하면 안 됩니다.

#### `qwen2.5:3b` 최적화 후 결과

`study-rag-llm:latest`에서 추론 출력 문제가 확인되어 답변 생성 모델을 `qwen2.5:3b`로 변경한 뒤 2026-09-22에 다시 측정했습니다. 모든 API가 200 응답을 반환했고, 영어 추론 내용 없이 정상적인 한국어 답변을 생성했습니다.

| API | 평균 | 중앙값 | 표준편차 | 최소 | 최대 | source 수 | 중복 수 | 결과 청크 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/chat` | 1.571초 | 1.522초 | 0.106초 | 1.508초 | 1.782초 | 3 | 0 | 14, 9, 15 |
| `/chat/filtered` | 1.507초 | 1.511초 | 0.011초 | 1.494초 | 1.518초 | 3 | 0 | 14, 9, 15 |
| `/chat/reranked` | 20.851초 | 21.355초 | 1.311초 | 18.496초 | 22.016초 | 2 | 0 | 14, 15 |
| `/chat/deduplicated` | 2.017초 | 1.558초 | 0.937초 | 1.527초 | 3.890초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-db` | 1.576초 | 1.573초 | 0.016초 | 1.558초 | 1.605초 | 3 | 0 | 14, 9, 15 |
| `/chat/deduplicated-semantic-reranked` | 23.737초 | 23.513초 | 0.937초 | 22.496초 | 24.851초 | 2 | 0 | 14, 15 |

당시 결과에서는 `/chat/filtered`가 가장 빠르고 안정적이며, `/chat/deduplicated-db`도 평균 1.576초와 표준편차 0.016초로 안정적이었습니다. 리랭킹 방식은 의미상 중복을 제거하지만 추가 Ollama 호출 때문에 20초 이상 걸렸습니다. 이후 운영 API는 의미 중복 제거 방식으로 변경되었고, 현재 운영 모델은 `qwen2.5:3b`입니다.

#### 라우터 분리 후 최신 결과

운영 API와 테스트 API를 분리한 뒤 `qwen2.5:3b` 기준으로 다시 측정했습니다. 운영 API `/chat`과 테스트 API 6개가 모두 200 응답을 반환했고, 모든 답변이 정상적으로 생성되었습니다.

| API | 평균 | 중앙값 | 표준편차 | 최소 | 최대 | Header | source 수 | 결과 청크 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/chat` | 1.572초 | 1.513초 | 0.116초 | 1.504초 | 1.803초 | 1.531초 | 3 | 14, 9, 15 |
| `/test/chat/basic` | 1.619초 | 1.579초 | 0.076초 | 1.537초 | 1.738초 | 1.536초 | 3 | 14, 9, 15 |
| `/test/chat/filtered` | 1.554초 | 1.555초 | 0.021초 | 1.527초 | 1.589초 | 1.557초 | 3 | 14, 9, 15 |
| `/test/chat/deduplicated` | 1.960초 | 1.541초 | 0.838초 | 1.535초 | 3.636초 | 1.534초 | 3 | 14, 9, 15 |
| `/test/chat/deduplicated-db` | 1.558초 | 1.557초 | 0.006초 | 1.551초 | 1.570초 | 1.553초 | 3 | 14, 9, 15 |
| `/test/chat/reranked` | 22.497초 | 22.877초 | 1.731초 | 19.688초 | 24.908초 | 22.875초 | 2 | 14, 15 |
| `/test/chat/semantic-reranked` | 22.905초 | 23.361초 | 2.001초 | 19.131초 | 24.997초 | 19.129초 | 2 | 14, 15 |

라우터 분리 직후 측정에서 `/chat`은 평균 1.572초로 안정적으로 동작했고, 테스트용 `/test/chat/filtered`와 성능 차이도 작았습니다. 당시 테스트 전략 중에서는 `/test/chat/deduplicated-db`가 평균 1.558초와 표준편차 0.006초로 가장 안정적이었습니다. 리랭킹 방식은 의미상 중복 청크 9를 제외하지만 추가 Ollama 호출 때문에 22초 이상 소요되었습니다.

#### 단계별 시간 측정 결과

라우터 분리 후 벤치마크에서 응답 헤더를 이용해 임베딩, DB 검색, 리랭킹, 답변 생성 시간을 확인했습니다. `Total`은 마지막 요청의 단계별 합계이며, `Average`는 5회 전체 HTTP 요청 시간의 평균입니다.

| API | 평균 전체 시간 | 임베딩 | DB | 리랭킹 | 답변 생성 | 결과 청크 |
|---|---:|---:|---:|---:|---:|---|
| `/chat` | 1.652초 | 0.303초 | 0.066초 | - | 1.187초 | 14, 9, 15 |
| `/test/chat/basic` | 1.551초 | 0.323초 | 0.066초 | - | 1.183초 | 14, 9, 15 |
| `/test/chat/filtered` | 1.581초 | 0.289초 | 0.069초 | - | 1.235초 | 14, 9, 15 |
| `/test/chat/deduplicated` | 2.152초 | 0.308초 | 0.069초 | - | 1.195초 | 14, 9, 15 |
| `/test/chat/deduplicated-db` | 1.581초 | 0.310초 | 0.047초 | - | 1.225초 | 14, 9, 15 |
| `/test/chat/reranked` | 20.565초 | 2.683초 | 0.063초 | 10.198초 | 5.849초 | 14, 15 |
| `/test/chat/semantic-reranked` | 20.682초 | 2.408초 | 0.091초 | 14.246초 | 5.981초 | 14, 15 |

단계별 결과에서 일반 검색 방식은 임베딩과 답변 생성이 대부분의 시간을 차지하고 DB 검색은 약 0.05~0.07초로 작습니다. 리랭킹 방식은 리랭킹 모델 호출이 추가되어 전체 시간이 약 20초까지 증가하므로 운영 API에는 사용하지 않습니다. `Total` 헤더는 마지막 요청 기준이므로 평균 성능 판단은 `Average` 열을 우선 사용합니다.

#### 최신 안정 실행 결과

의미 중복 제거를 운영 API에 적용한 뒤 품질 테스트와 벤치마크를 순서대로 실행했습니다. 품질 테스트 4개와 벤치마크 전체 요청이 모두 성공했습니다.

| 항목 | 결과 |
|---|---|
| 품질 테스트 | 6개 모두 PASS |
| 문서 관리 회귀 테스트 | 업로드·조회·교체·삭제 전체 PASS |
| 벤치마크 반복 횟수 | API별 5회 |
| HTTP 상태 | 전체 200 |
| 운영 API 평균 응답 시간 | 1.653초 |
| 운영 API 답변 생성 시간 | 1.197초 |
| 운영 API DB 검색 시간 | 0.077초 |
| 운영 API 출처 | 21, 22 |
| 운영 API 중복 출처 | 0개 |

최신 실행에서 `/chat`은 평균 1.653초에 출처 2개를 반환했습니다. `/test/chat/semantic-deduplicated`는 평균 1.688초로 확인되어 운영 API 전략과 유사한 성능을 보였습니다. `/test/chat/reranked`와 `/test/chat/semantic-reranked`는 각각 평균 28.134초와 29.436초로 추가 Ollama 호출 비용이 커서 테스트 전용으로 유지합니다.

현재 결과만 기준으로 한 임시 판단:

- 품질 테스트 기준: `/chat`과 테스트 API 모두 정상 답변 확인
- 응답 시간과 품질의 균형: `/chat`
- 운영 기본 전략: `/chat`의 `semantic-deduplicated`
- 빠른 의미 중복 제거 실험: `/test/chat/semantic-deduplicated`
- 리랭킹: 추가 지연이 크므로 테스트 전용 유지

기존 모델 기준 측정에서 `/chat/reranked`와 `/chat/deduplicated-semantic-reranked`는 의미상 중복 청크 9를 제외했습니다. 두 방식 모두 추가 Ollama 호출로 느렸고, 의미 중복 제거를 적용한 방식이 더 높은 평균 지연 시간을 보였습니다. 문자열 기반 중복 제거 방식은 표현이 다른 청크를 동일 내용으로 판단하지 못했습니다.

기존 모델 기준 추천 전략:

| 용도 | 추천 API | 이유 |
|---|---|---|
| 기본 서비스 | `/chat` | `semantic-deduplicated` 전략을 사용하는 운영 API |
| 빠른 의미 중복 제거 실험 | `/test/chat/semantic-deduplicated` | 추가 Ollama 호출 없이 중복 출처를 줄임 |
| 리랭킹 실험 | `/test/chat/reranked` | 출처를 줄이지만 응답 시간이 큼 |
| 비교·실험용 | `/test/chat/*` | 검색 전략 차이 분석용 |

현재 한계:

- TXT의 빈 줄 문단과 명시적 제목 결합을 지원하며, 700자 초과 문단은 추가 분할
- PDF·DOCX는 아직 지원하지 않음
- 고정 거리 임계값 `0.5` 검증 필요
- 의미 거리와 문자열 포함 관계 기반 중복 제거를 지원하며 임계값 추가 검증 필요
- 단계별 성능 측정 결과를 파일로 저장하는 기능 부족
- SQLAlchemy 비동기 세션을 사용하며, 서비스 쓰기 작업의 트랜잭션 경계를 적용함
- 인증, 악성 파일 검사, 운영 모니터링은 다음 단계

테스트용 ESS 데이터는 학습용 가상 데이터이며 실제 장비 운영 기준으로 사용하면 안 됩니다.

## 12. 현재 진행 상황

| 기능 | 상태 | 비고 |
|---|---|---|
| TXT 업로드와 청크 분할 | 완료 | 제목은 다음 본문과 결합하고, 문단은 유지하며 700자 초과 문단은 100자 overlap으로 추가 분할 |
| 임베딩 생성과 저장 | 완료 | Ollama `embeddinggemma` |
| 기본 유사도 검색 | 완료 | 상위 3개 |
| 거리 필터링 | 완료 | 기준 `0.5` |
| Ollama 리랭킹 | 완료 | `qwen3:4b` |
| Python 중복 제거 | 완료 | `/test/chat/deduplicated` |
| DB 중복 제거 | 완료 | `/test/chat/deduplicated-db` |
| 전체 응답 시간 측정 | 완료 | `X-Process-Time` |
| 단계별 성능 측정 | 완료 | `X-Embedding-Time`, `X-DB-Time`, `X-Rerank-Time`, `X-Answer-Time` |
| 검색 로직 공통화 | 완료 | `core/util/retrieval.py`의 `find_documents_by_question()`로 검색 전략 통합 |
| 의미상 중복 제거 | 완료 | `/chat`, `/test/chat/semantic-deduplicated` |
| 의미상 중복 제거·리랭킹 | 완료 | `/test/chat/semantic-reranked` |
| 추론 답변 모델 설치 | 완료 | `study-rag-llm:latest` 설치 확인 |
| 추론 답변 모델 코드 전환 | 보류 | 운영 모델은 `qwen2.5:3b`, 추론 모델은 실험용 |
| 테스트·운영 라우터 분리 | 완료 | `/test/chat/*`와 `/chat` 구조 |
| 검색 품질 자동 평가 | 완료 | `test/test_quality.py` 6개 케이스 통과 |
| 검색 벤치마크 | 완료 | `test/test_search_benchmark.py`, API별 5회 반복 측정 |
| 벤치마크 중복 실행 방지 | 완료 | 잠금 파일과 진행 상태 출력 적용 |
| 자동화 테스트 | 완료 | 품질·문서·오류·장애·동시성 테스트 완료 |
| 계층형 아키텍처 | 완료 | `router → service → repo` 구조로 분리 |
| SQLAlchemy·asyncpg 전환 | 완료 | ORM 모델·AsyncSession·Repository와 `postgresql+asyncpg` 적용 |
| Singleton·Factory DI 컨테이너 | 완료 | 인프라 Singleton, Service·Repository는 요청별 Factory와 Depends 캐시 |
| CBV 라우터 | 완료 | 클래스 속성 주입, 기존 API 경로 유지 |
| 여러 서비스의 원자적 변경 | 격리 테스트 완료 | 문서·청크 공동 롤백과 취소 시 정리 검증 |
| 서비스 트랜잭션 | 완료 | 문서 생성·업로드·교체·삭제의 DB 변경을 서비스 흐름에서 처리 |

## 13. 남은 작업 목록

| 단계 | 작업 | 우선순위 | 상태 | 비고 |
|---|---|---:|---|---|
| 1단계 | PostgreSQL/pgvector 컨테이너와 포트 확인 | 높음 | 완료 | Docker Desktop, 로컬 포트 25433 확인 |
| 1단계 | 테이블 구조·벡터 차원·인덱스 확인 | 높음 | 완료 | `vector(768)`, HNSW·IVFFlat 확인 |
| 1단계 | FastAPI 실제 실행 명령과 포트 확인 | 높음 | 완료 | `--env-file .env.dev`, 포트 8000 확인 |
| 1단계 | 테스트·운영 라우터 분리 | 높음 | 완료 | `/test/chat/*`와 `/chat` 구조로 이동 |
| 1단계 | `study-rag-llm:latest`를 답변 생성에 연결 | 높음 | 보류 | 추론 출력과 응답 시간이 불안정해 실험용 유지 |
| 2단계 | 검색 SQL과 결과 변환 로직 공통화 | 높음 | 완료 | `repo/document_repo.py`로 이동 |
| 2단계 | 검색 전략을 공통 함수 옵션으로 통합 | 높음 | 완료 | `basic`, `filtered`, `deduplicated`, `deduplicated-db` 지원 |
| 2단계 | 운영용 기본 전략을 `/chat`에 연결 | 높음 | 완료 | `/chat`에 `semantic-deduplicated` 전략 연결 |
| 2단계 | 관련성 낮은 결과의 답변 생성 중단 기준 검증 | 높음 | 부분 완료 | 거리 임계값 실험 필요 |
| 3단계 | 질문 유형별 테스트와 기대 출처 작성 | 높음 | 완료 | `test/test_quality.py` 6개 평가 케이스 통과 |
| 3단계 | 검색 방식별 정확도·재현율·응답 시간 비교 자동화 | 높음 | 부분 완료 | 5회 반복 시간 비교 완료, 질문 세트 확장 필요 |
| 3단계 | `study-rag-llm:latest` 기준 전체 벤치마크 | 높음 | 완료 | 추론 출력 문제와 높은 응답 시간 확인 |
| 3단계 | 새 모델 추론 시간 최적화 | 높음 | 보류 | `qwen3:4b` 기반 모델의 `think: false` 동작 문제 해결 필요 |
| 3단계 | `qwen2.5:3b` 운영 모델 후보 확정 | 높음 | 완료 | 정상 답변과 평균 1~2초대 응답 확인 |
| 3단계 | 의미상 중복 청크 제거 기준 검토 | 중간 | 완료 | 거리 기준 `0.1`과 문자열 포함 관계 적용, 6개 질문 검증 |
| 3단계 | 문서에 없는 질문의 거절 응답 테스트 | 높음 | 완료 | 출처 빈 목록과 거절 문구 검증 |
| 4단계 | 단계별 성능 측정 추가 | 중간 | 완료 | 임베딩·DB·리랭킹·생성 시간 응답 헤더 제공 |
| 4단계 | SQLAlchemy 비동기 DB 호출 | 높음 | 완료 | `AsyncEngine`, `AsyncSession`, Repository 적용 |
| 4단계 | 운영 요청 로그 추가 | 중간 | 완료 | 메서드·경로·상태 코드·처리 시간 기록 |
| 4단계 | 자동화 테스트 추가 | 높음 | 완료 | 문서·오류·장애·동시성·품질 테스트 완료 |
| 4단계 | 최종 패키지·파일 구조 정리 | 중간 | 완료 | 단수형 이름, `core/llm`, `core/util`, 계층별 파일명 적용 |
| 4단계 | 서비스 트랜잭션 경계 정리 | 높음 | 완료 | 문서 생성·업로드·조회·교체·삭제를 Service에서 처리 |
| 4단계 | 동시성 병목 상세 분석 | 중간 | 보류 | 기본 3개 동시 요청 검증만 완료 |
| 5단계 | 토큰 기준 청크와 overlap 적용 | 중간 | 미완료 | 현재는 문자 수 기준 700자와 100자 overlap |
| 5단계 | PDF·DOCX 지원 | 낮음 | 미완료 | TXT 안정화 후 진행 |
| 6단계 | 대화 문맥 유지와 웹 UI 구현 | 중간 | 미완료 | 서비스 기능 확장 |
| 6단계 | 인증·악성 파일 검사·배포 구성 | 높음 | 미완료 | 운영 전 필요 |
| 6단계 | API Key 인증 | 높음 | 미완료 | `/chat`과 문서 관리 API 보호 필요 |

## 14. 다음 진행 순서

1. 서버 재시작 후 실제 PostgreSQL·Ollama 환경에서 품질·문서 관리 회귀 테스트 확인
2. API Key 인증과 인증 실패 회귀 테스트 추가
3. 운영 모니터링과 로그 보존 정책 검토
4. 동시 업로드 중복 방지를 위한 DB 유일성 정책·마이그레이션 검토
5. 악성 파일 내용 검사와 업로드 저장 정책 검토
6. 필요 시 토큰 기준 청크 분할과 PDF·DOCX 지원 추가

### 이번 구조 변경 검증 범위

추가 코드 검수에서 Repository의 불필요한 실행 래퍼와 모델 별칭을 제거하고 세션 사용을 직접 드러내도록 정리했습니다. 세션 생성은 Container의 요청 의존성 한 곳에서 관리합니다. DB URL은 SQLAlchemy `URL.create()`로 구성하여 비밀번호의 특수문자를 안전하게 처리합니다.

공백 없는 긴 문서의 청크 누락과 overlap=0일 때 본문 건너뛰기를 수정했습니다. 업로드 본문은 제한 용량+1바이트까지만 읽어서 추가 메모리 사용을 제한합니다(HTTP 요청 수신 자체의 크기 제한은 프록시 등에서 별도 설정 필요). 비객체 JSON 리랭킹 응답 방어와 DB 종료 실패 시 Container 초기화도 보강했습니다. 기존 저장 청크는 자동 변경하지 않으며, 누락 영향을 받은 문서는 필요 시 교체 API로 재처리해야 합니다.

이 경계값을 검증하는 단위 테스트 7개가 통과했습니다.

```powershell
uv run python -m unittest discover -s test -p test_unit.py -v
```

`test/test_architecture.py`의 8개 테스트가 통과했습니다. SQLite 격리 DB와 모킹을 사용해 요청별 세션 분리, 요청 내 객체 재사용, 문서 CRUD, 하위 서비스 실패·취소 시 공동 롤백, 교체 실패 시 원본 보존, API 경로 및 자원 종료를 확인했습니다. 실제 PostgreSQL 벡터 검색과 Ollama 품질·성능을 재측정한 결과는 아닙니다. 위 벤치마크 수치는 이전 구조의 기록입니다.

```powershell
uv run python -m unittest discover -s test -p test_architecture.py -v
```

모킹 기반 장애 테스트도 DB 503, Ollama 연결 실패 503, 응답 시간 초과 504, 연결 풀 대기 초과 503을 검증했습니다.

```powershell
uv run python test/test_fault_api.py
```

검색 로직 공통화, 테스트·운영 API 분리, 의미 중복 제거, 품질 테스트, 문서 관리 테스트와 기본 동시성 검증은 완료했습니다. 동시성 병목 상세 분석은 보류하고, 다음 개발 우선순위는 API 인증과 운영 안전성 보강입니다.

## 15. 확인 필요 항목

- Docker Desktop의 실제 컨테이너명, 포트, 이미지
- PostgreSQL/pgvector 테이블 생성 SQL 원본 파일
- HNSW와 IVFFlat 중 실제 대규모 데이터 운영 시 우선 사용할 인덱스
- FastAPI 실제 실행 포트: 8000 기준 확인 완료
- Ollama 모델 설치 상태
- `study-rag-llm:latest`의 모델 호출별 추론 시간과 생성 토큰 수
- `study-rag-llm:latest`의 추론 시간 개선 후 재검증 여부
- 자동화 테스트 질문과 기대 출처 확장
- 단계별 성능 로그 저장 방식
- 문서 교체 API의 DB 무결성 검증
