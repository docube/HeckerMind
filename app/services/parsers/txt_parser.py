# app/services/parsers/txt_parser.py

def parse_txt(file_path: str) -> str:
    """
    Extracts text from a plain TXT file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()
