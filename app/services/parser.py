# app/services/parser.py

import os

from app.services.parsers.pdf_parser import parse_pdf
from app.services.parsers.docx_parser import parse_docx
from app.services.parsers.xlsx_parser import parse_xlsx
from app.services.parsers.csv_parser import parse_csv
from app.services.parsers.txt_parser import parse_txt
from app.services.parsers.pptx_parser import parse_pptx

def parse_document(file_path: str) -> str:
    """
    Main parsing dispatcher. Receives a file path, detects the extension,
    and routes it to the appropriate parser.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return parse_xlsx(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".pptx":
        return parse_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
