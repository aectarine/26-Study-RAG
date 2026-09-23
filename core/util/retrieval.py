"""검색 관련 호환 유틸리티입니다."""

from core.dep.container import container


async def find_documents_by_question(
    question: str,
    strategy: str = "basic",
    timings=None,
):
    """DI 컨테이너를 통해 문서를 검색합니다.

    라우터와 서비스 계층 외부에서 기존 검색 함수를 직접 호출해야 하는
    학습 코드나 테스트를 위해 제공하는 얇은 호환 함수입니다.
    """
    async with container.database().session_factory() as session:
        return await container.rag_service(
            session=session,
            repository=container.document_repository(session=session),
        ).find_documents_by_question(
            question,
            strategy,
            timings,
        )
