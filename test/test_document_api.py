import sys
from uuid import uuid4

import httpx


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 180.0


def request_file(filename: str, content: str) -> dict:
    return {
        "file": (
            filename,
            content.encode("utf-8"),
            "text/plain"
        )
    }


def check_status(response: httpx.Response, expected_status: int) -> None:
    if response.status_code != expected_status:
        raise AssertionError(
            f"HTTP {response.status_code} 응답: {response.text}"
        )


def run_document_api_test() -> list[str]:
    filename = f"__document_api_test_{uuid4().hex}.txt"
    original_content = (
        "업로드 문서 관리 API 테스트용 첫 번째 문단입니다.\n\n"
        "업로드 문서 업로드와 청크 저장을 확인합니다."
    )
    replaced_content = (
        "업로드 문서 관리 API 교체 테스트용 내용입니다.\n\n"
        "업로드 문서 교체 후 새로운 청크가 저장되는지 확인합니다."
    )
    source_document_id = None
    rs = []

    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = client.post(
                "/api/documents/upload",
                files=request_file(filename, original_content)
            )
            check_status(response, 200)
            upload_rs = response.json()
            source_document_id = upload_rs["source_document_id"]

            if upload_rs["status"] != "saved":
                raise AssertionError("최초 업로드 상태가 saved가 아닙니다.")

            if upload_rs["chunk_count"] < 1:
                raise AssertionError("최초 업로드 청크가 생성되지 않았습니다.")

            rs.append("업로드")

            response = client.get("/api/documents")
            check_status(response, 200)
            documents = response.json()["documents"]

            if not any(
                    document["source_document_id"] == source_document_id
                    for document in documents
            ):
                raise AssertionError("업로드 문서 목록에서 업로드 문서를 찾지 못했습니다.")

            rs.append("목록 조회")

            response = client.get(f"/api/documents/{source_document_id}")
            check_status(response, 200)
            detail = response.json()

            if detail["chunk_count"] != upload_rs["chunk_count"]:
                raise AssertionError("상세 조회의 청크 수가 업로드 결과와 다릅니다.")

            rs.append("상세 조회")

            response = client.post(
                "/api/documents/upload",
                files=request_file(filename, original_content)
            )
            check_status(response, 200)

            if response.json()["status"] != "skipped":
                raise AssertionError("동일 업로드 문서 재업로드가 skipped가 아닙니다.")

            rs.append("동일 업로드 문서 중복 확인")

            response = client.put(
                f"/api/documents/{source_document_id}",
                files=request_file(filename, replaced_content)
            )
            check_status(response, 200)
            replace_rs = response.json()

            if replace_rs["status"] != "replaced":
                raise AssertionError("업로드 문서 교체 상태가 replaced가 아닙니다.")

            rs.append("업로드 문서 교체")

            response = client.delete(f"/api/documents/{source_document_id}")
            check_status(response, 200)
            delete_rs = response.json()

            if delete_rs["status"] != "deleted":
                raise AssertionError("업로드 문서 삭제 상태가 deleted가 아닙니다.")

            rs.append("업로드 문서 삭제")

            response = client.get(f"/api/documents/{source_document_id}")
            check_status(response, 404)
            rs.append("삭제 후 조회 차단")

    finally:
        if source_document_id is not None:
            try:
                with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
                    client.delete(f"/api/documents/{source_document_id}")
            except httpx.RequestError:
                pass

    return rs


if __name__ == "__main__":
    try:
        completed_steps = run_document_api_test()
        print("[PASS] 업로드 문서 관리 API 테스트")
        print(f"검증 항목: {', '.join(completed_steps)}")
    except (AssertionError, httpx.RequestError) as error:
        print("[FAIL] 업로드 문서 관리 API 테스트")
        print(f"error: {error}")
        sys.exit(1)
