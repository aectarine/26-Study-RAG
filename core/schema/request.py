from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    text: str


class DocumentRequest(BaseModel):
    content: str


class SearchRequest(BaseModel):
    question: str
