from pydantic import BaseModel


class SourceDocumentResponse(BaseModel):
    id: int
    content: str
    distance: float
    source_document_id: int | None
    filename: str | None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocumentResponse]
