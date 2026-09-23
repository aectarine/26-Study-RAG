import hashlib
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile


MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def split_text(content: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars는 양수이고 overlap은 0 이상 max_chars 미만이어야 합니다.")
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []

    paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
    heading_pattern = re.compile(r"^(?:#{1,6}\s+|\d+[.)]\s+).+")
    merged = []
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        if (
                heading_pattern.match(paragraph)
                and len(paragraph) <= 120
                and index + 1 < len(paragraphs)):
            merged.append(f"{paragraph}\n{paragraphs[index + 1]}")
            index += 2
        else:
            merged.append(paragraph)
            index += 1

    chunks = []
    for paragraph in merged:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            target_end = min(start + max_chars, len(paragraph))
            end = target_end

            if target_end < len(paragraph):
                minimum = start + max_chars // 2
                part = paragraph[start:target_end]
                matches = list(re.finditer(r"[.!?](?=\s|$)", part))
                if matches and start + matches[-1].end() >= minimum:
                    end = start + matches[-1].end()
                else:
                    boundary = paragraph.rfind(" ", minimum, target_end)
                    if boundary > start:
                        end = boundary

            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                break

            next_start = max(end - overlap, start + 1)
            # 단어 경계 보정은 이미 저장한 구간 안에서만 수행합니다.
            # 공백 없는 문서에서도 end 이후의 본문을 건너뛰지 않습니다.
            boundary = paragraph.find(" ", next_start, end)
            start = boundary + 1 if boundary >= 0 else next_start

    return chunks


def create_content_hash(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def read_text_upload(file: UploadFile) -> str:
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(400, "TXT 파일만 업로드할 수 있습니다.")
    if Path(file.filename).name != file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(400, "파일명에 경로를 포함할 수 없습니다.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "업로드 파일은 5MB 이하만 허용됩니다.")
    try:
        content = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise HTTPException(
            400, "UTF-8로 인코딩된 TXT 파일을 업로드해 주세요."
        ) from error
    if not content:
        raise HTTPException(400, "파일 내용이 비어 있습니다.")
    return content
