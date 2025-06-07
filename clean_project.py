# clean_project.py

import os
import shutil

folders_to_clean = ["uploads", "vectorstore"]

for folder in folders_to_clean:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"[🧹] Cleaned: {folder}")
    os.makedirs(folder, exist_ok=True)
    print(f"[📁] Recreated: {folder}")