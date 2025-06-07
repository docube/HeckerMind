# app/services/parsers/csv_parser.py

import pandas as pd

def parse_csv(file_path: str) -> str:
    """
    Extracts text from a CSV file.
    Cleans and formats it nicely.
    """
    extracted_text = ""

    df = pd.read_csv(file_path)

    # Drop fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if not df.empty:
        for index, row in df.iterrows():
            row_text = " | ".join(str(cell).strip() for cell in row)
            extracted_text += row_text + "\n"

    return extracted_text.strip()
