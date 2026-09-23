import json
import time

import httpx

from core.cfg.settings import Settings


class OllamaClient:
    """Ollama 답변 생성과 검색 결과 리랭킹을 담당하는 클라이언트입니다."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self.settings = settings
        self._http_client = http_client

    async def generate_answer(
            self, question: str, documents: list[dict],
            timings: dict[str, float] | None = None) -> str:
        context = "\n\n".join(
            f"[문서 {doc['id']}]\n{doc['content']}"
            for doc in documents
        )
        prompt = f"""
문서 기반 질의응답 시스템입니다.

규칙:
- 아래 문서에 있는 내용만 사용하세요.
- 문서에 답이 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답하세요.
- 한국어로만 답변하세요.
- 추론 과정 없이 최종 답변만 간결하게 작성하세요.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
"""

        started_at = time.perf_counter()
        response = await self._http_client.post(
            f"{self.settings.ollama_url}/api/generate",
            timeout=httpx.Timeout(self.settings.answer_timeout, pool=self.settings.http_pool_timeout),
            json={
                "model": self.settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "top_p": 0.8,
                    "num_predict": 128,
                },
                "keep_alive": "10m",
            },
        )
        response.raise_for_status()
        rs = response.json()

        if timings is not None:
            timings["answer-time"] = time.perf_counter() - started_at

        _print_metrics("답변 생성 Ollama 성능", rs)
        answer = rs["response"]
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1]
        return answer.strip()

    async def rerank_documents(
            self, question: str, documents: list[dict],
            timings: dict[str, float] | None = None) -> list[dict]:
        if not documents:
            return []

        candidates = [
            {"id": document["id"], "content": document["content"]}
            for document in documents
        ]
        prompt = f"""
당신은 RAG 검색 결과를 선별하는 역할입니다.

사용자 질문에 답변하는 데 직접 필요한 정보가 포함된 청크만 선택하세요.

규칙:
1. 질문과 주제가 비슷하다는 이유만으로 선택하지 마세요.
2. 질문에 답변하는 데 필요한 정보가 포함된 청크를 선택하세요.
3. 여러 청크의 정보를 조합해야 한다면 필요한 청크를 모두 선택하세요.
4. 답변에 필요한 정보가 없다면 빈 배열을 반환하세요.
5. 아래 JSON 형식으로만 응답하세요. 설명은 작성하지 마세요.

응답 형식:
{{"selected_ids": [7, 8]}}

사용자 질문:
{question}

검색된 청크:
{json.dumps(candidates, ensure_ascii=False)}
"""

        started_at = time.perf_counter()
        response = await self._http_client.post(
            f"{self.settings.ollama_url}/api/generate",
            timeout=httpx.Timeout(self.settings.rerank_timeout, pool=self.settings.http_pool_timeout),
            json={
                "model": self.settings.rerank_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 64,
                },
                "keep_alive": "10m",
            },
        )
        response.raise_for_status()
        rs = response.json()

        if timings is not None:
            timings["rerank-time"] = time.perf_counter() - started_at

        _print_metrics("리랭킹 Ollama 성능", rs)
        raw_response = rs.get("response", "")

        try:
            rerank_rs = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"리랭킹 JSON 파싱 실패: {raw_response}")
            return []

        if not isinstance(rerank_rs, dict):
            return []
        selected_ids = rerank_rs.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            print(f"리랭킹 selected_ids 형식 오류: {selected_ids}")
            return []

        available_ids = {document["id"] for document in documents}
        normalized_ids = set()
        for selected_id in selected_ids:
            try:
                normalized_ids.add(int(selected_id))
            except (TypeError, ValueError):
                continue

        selected_ids = normalized_ids & available_ids
        return [
            document
            for document in documents
            if document["id"] in selected_ids
        ]


def _print_metrics(prefix: str, rs: dict) -> None:
    metrics = {
        "total_duration": rs.get("total_duration"),
        "load_duration": rs.get("load_duration"),
        "prompt_eval_duration": rs.get("prompt_eval_duration"),
        "eval_duration": rs.get("eval_duration"),
        "eval_count": rs.get("eval_count"),
    }
    print(f"{prefix}: {metrics}")
