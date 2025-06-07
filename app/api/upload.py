# app/api/upload.py

from typing import List
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.models.upload_models import UploadResponse, UploadResult
from app.ingestion.document_ingestor import DocumentIngestor

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".txt"}
UPLOAD_DIR = "uploads"

def clean_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    name = name.strip().replace(" ", "_").lower()
    return f"{name}{ext.lower()}"

@router.post("/upload", response_model=UploadResult)
async def upload_files(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ingestion_results = []
    ingestor = DocumentIngestor()

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type {ext} not allowed.")

        clean_name = clean_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, clean_name)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            result = ingestor.ingest(file_path)
            ingestion_results.append(UploadResponse(
                file=os.path.basename(file_path),
                status="success",
                message=result
            ))
        except Exception as e:
            ingestion_results.append(UploadResponse(
                file=os.path.basename(file_path),
                status="error",
                message=str(e)
            ))

    return UploadResult(
        message="Upload and ingestion completed!",
        results=ingestion_results
    )

@router.get("/documents", summary="List all uploaded documents")
async def list_documents():
    if not os.path.exists("uploads"):
        return JSONResponse(content={"documents": []})

    files = os.listdir("uploads")
    documents = [file for file in files if os.path.isfile(os.path.join("uploads", file))]

    return {"documents": documents}
