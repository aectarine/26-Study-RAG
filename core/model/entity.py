from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def get_col_spec(self, **kwargs):
        return f"vector({self.dimensions})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None or isinstance(value, str):
                return value
            return str(value)

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            # pgvector는 '[0.1,0.2,...]' 형태의 문자열로 값을 돌려줍니다.
            if value is None or not isinstance(value, str):
                return value
            body = value.strip().strip("[]")
            if not body:
                return []
            return [float(item) for item in body.split(",")]

        return process


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    inserted: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    chunks: Mapped[list["RagDocument"]] = relationship(
        back_populates="source_document", cascade="all, delete-orphan"
    )


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    source_document: Mapped[SourceDocument | None] = relationship(
        back_populates="chunks"
    )
