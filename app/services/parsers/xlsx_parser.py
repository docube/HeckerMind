# app/services/parsers/xlsx_parser.py

import pandas as pd

def parse_xlsx(file_path: str) -> str:
    """
    Extracts text from XLSX and XLS files.
    Reads all sheets and formats data cleanly.
    """
    extracted_text = ""

    # Read all sheets
    xls = pd.read_excel(file_path, sheet_name=None)

    for sheet_name, df in xls.items():
        extracted_text += f"--- Sheet: {sheet_name} ---\n"
        
        # Drop fully empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        
        if not df.empty:
            for index, row in df.iterrows():
                row_text = " | ".join(str(cell).strip() for cell in row)
                extracted_text += row_text + "\n"
        
        extracted_text += "\n"  # Separator between sheets

    return extracted_text.strip()
