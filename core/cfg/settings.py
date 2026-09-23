from functools import lru_cache
import os

from dotenv import load_dotenv
from sqlalchemy.engine import URL


load_dotenv()


class Settings:
    """시작 시 읽어서 애플리케이션 전체에서 공유하는 설정입니다."""

    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "rag_db")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.db_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "0"))
        self.db_pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "10"))
        self.http_max_connections = int(os.getenv("HTTP_MAX_CONNECTIONS", "2"))
        self.http_pool_timeout = float(os.getenv("HTTP_POOL_TIMEOUT", "5"))
        if (self.db_pool_size < 1 or self.db_max_overflow < 0
                or self.db_pool_timeout <= 0 or self.http_max_connections < 1
                or self.http_pool_timeout <= 0):
            raise ValueError("연결 풀 크기와 대기 시간은 양수, DB 추가 연결 수는 0 이상이어야 합니다.")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.llm_model = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.rerank_model = os.getenv("RERANK_MODEL", "qwen3:4b")
        self.answer_timeout = float(os.getenv("OLLAMA_ANSWER_TIMEOUT", "180"))
        self.rerank_timeout = float(os.getenv("OLLAMA_RERANK_TIMEOUT", "120"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "embeddinggemma")

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+asyncpg", username=self.db_user, password=self.db_password,
            host=self.db_host, port=int(self.db_port), database=self.db_name,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
