from core.llm.ollama import OllamaClient

NO_ANSWER_MESSAGE = "제공된 문서에서 관련 정보를 찾을 수 없습니다."


class ChatService:
    def __init__(self, rag_service, ollama_client: OllamaClient):
        self.rag_service = rag_service
        self.ollama_client = ollama_client

    @staticmethod
    def _build_response(question: str, answer: str, documents: list[dict]) -> dict:
        """거절 답변이면 출처를 비웁니다."""
        found = not answer.strip().startswith(NO_ANSWER_MESSAGE)
        return {"question": question, "answer": answer,
                "sources": documents if found else []}

    async def generate_answer_by_question(self, question, strategy, timings):
        documents = await self.rag_service.find_documents_by_question(
            question, strategy, timings
        )
        if not documents:
            return self._build_response(question, NO_ANSWER_MESSAGE, [])
        answer = await self.ollama_client.generate_answer(question, documents, timings)
        return self._build_response(question, answer, documents)

    async def generate_reranked_answer_by_question(
            self, question, timings, semantic=False):
        strategy = "semantic-deduplicated" if semantic else "filtered"
        documents = await self.rag_service.find_documents_by_question(
            question, strategy, timings
        )
        selected = await self.ollama_client.rerank_documents(question, documents, timings)
        if not selected:
            return self._build_response(
                question, "등록된 문서에서 질문에 답할 수 있는 정보를 찾지 못했습니다.", []
            )
        answer = await self.ollama_client.generate_answer(question, selected, timings)
        return self._build_response(question, answer, selected)
