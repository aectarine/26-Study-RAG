import json

import httpx

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"


async def generate_answer(question: str, documents: list[dict]) -> str:
    """
    검색된 문서를 참고하여 AI 답변을 생성합니다.
    """

    # 1. 검색된 문서를 하나의 문자열로 구성
    context = "\n\n".join(
        f"[문서 {doc['id']}]\n{doc['content']}"
        for doc in documents
    )
    print(f"context: \n{context}")

    # 2. LLM에 전달할 프롬프트 생성
    prompt = f"""
당신은 ESS 배터리 관련 문서 기반 질의응답 AI입니다.

아래 제공된 문서만 참고하여 사용자의 질문에 답변하세요.

규칙:
1. 제공된 문서에 근거하여 답변하세요.
2. 문서에 없는 내용은 임의로 만들어내지 마세요.
3. 문서에서 답을 찾을 수 없다면
   "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.
4. 한국어로만 답변하세요.
5. 추론 과정이나 분석 내용은 출력하지 마세요.
6. 최종 답변만 간결하게 출력하세요.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
"""

    # 3. OLLAMA LLM 호출
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False
            }
        )
        response.raise_for_status()
        data = response.json()

        # 4. AI가 생성한 응답
        answer = data["response"]

        # 5. 추론 내용이 포함된 경우 제거
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1]

        # 6. 앞뒤 공백 제거
        return answer.strip()


async def rerank_documents(question: str, documents: list[dict]) -> list[dict]:
    if not documents:
        return []

    candidates = [
        {
            "id": document["id"],
            "content": document["content"],
        }
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:4b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False
            }
        )

        response.raise_for_status()
        result = response.json()

    selected_ids = json.loads(result["response"])["selected_ids"]

    # 모델이 반환한 ID 중 실제 검색 결과에 존재하는 ID만 사용
    selected_ids = set(selected_ids)

    return [
        document
        for document in documents
        if document["id"] in selected_ids
    ]
