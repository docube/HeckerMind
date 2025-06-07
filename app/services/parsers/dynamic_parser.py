# app/services/parsers/dynamic_parser.py

import os
from app.services.parsers import (
    parse_pdf,
    parse_docx,
    parse_xlsx,
    parse_csv,
    parse_ppt,
    parse_txt
)

def parse_document(file_path: str) -> str:
    """
    Detects file type and uses appropriate parser to extract text.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return parse_xlsx(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".pptx":
        return parse_ppt(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
