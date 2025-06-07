# app/models/upload_models.py

from pydantic import BaseModel
from typing import List

class UploadResponse(BaseModel):
    file: str
    status: str
    message: str

class UploadResult(BaseModel):
    message: str
    results: List[UploadResponse]

class SearchResult(BaseModel):
    query: str
    results: List[str]
