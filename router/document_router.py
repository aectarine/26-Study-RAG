from fastapi import APIRouter, Depends, File, UploadFile
from fastapi_utils.cbv import cbv

from core.dep.container import get_document_service
from core.schema.request import DocumentRequest
from core.util.document import read_text_upload
from service.document_service import DocumentService


router = APIRouter(tags=["업로드 문서 관리"])


@cbv(router)
class DocumentRouter:
    service: DocumentService = Depends(get_document_service)

    @router.post("/")
    async def create_document(self, body: DocumentRequest):
        saved = await self.service.create_document(body.content)
        return {**saved, "content": body.content, "status": "saved"}


    @router.post("/upload")
    async def upload_document(self, file: UploadFile = File(...)):
        content = await read_text_upload(file)
        return await self.service.upload_document(file.filename, content)


    @router.get("/")
    async def find_all_documents(self):
        documents = await self.service.find_all_documents()
        return {"total": len(documents), "documents": documents}


    @router.get("/{source_document_id}")
    async def find_document_by_id(self,
            source_document_id: int):
        return await self.service.find_document_by_id(source_document_id)


    @router.put("/{source_document_id}")
    async def replace_document_by_id(self,
            source_document_id: int, force: bool = False,
            file: UploadFile = File(...)):
        content = await read_text_upload(file)
        return await self.service.replace_document_by_id(
            source_document_id, file.filename, content, force
        )


    @router.delete("/{source_document_id}")
    async def delete_document_by_id(self,
            source_document_id: int):
        return await self.service.delete_document_by_id(source_document_id)
