# app/services/parsers/pdf_parser.py

# ✅ Updated: pdf_parser.py with OCR fallback support

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os


def extract_text_from_pdf_with_ocr(file_path: str) -> str:
    try:
        images = convert_from_path(file_path)
        text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            print(f"[OCR] Page {i + 1}: {len(page_text)} characters")
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return ""


def parse_pdf(file_path: str) -> str:
    """
    Extracts text (including tables) from a PDF file using pdfplumber.
    Falls back to OCR if no text is found.
    """
    extracted_text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                extracted_text += page_text + "\n"

                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(cell if cell else "" for cell in row)
                        extracted_text += row_text + "\n"
                extracted_text += "\n"
    except Exception as e:
        print(f"[ERROR] Failed to read PDF with pdfplumber: {e}")

    if not extracted_text.strip():
        print("[🔍] No text found in PDF — attempting OCR fallback...")
        return extract_text_from_pdf_with_ocr(file_path)

    return extracted_text.strip()

