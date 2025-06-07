# app/ingestion/vectorstore.py

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config.settings import get_settings
import faiss
import pickle
import numpy as np

settings = get_settings()

class VectorStoreManager:
    def __init__(self, base_path: str = "vectorstore/"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    def save_vectorstore(self, embeddings: List[List[float]], texts: List[str], document_name: str):
        if not embeddings or not texts:
            raise ValueError("Embeddings and texts must not be empty.")
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must be the same length.")

        vectors = np.array(embeddings).astype("float32")
        dim = vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        os.makedirs(self.base_path, exist_ok=True)
        doc_folder = os.path.join(self.base_path, document_name)
        os.makedirs(doc_folder, exist_ok=True)

        index_path = os.path.join(doc_folder, "index.faiss")
        metadata_path = os.path.join(doc_folder, "index.pkl")

        faiss.write_index(index, index_path)
        print(f"[VERIFY] FAISS index file exists: {os.path.exists(index_path)}")
        with open(metadata_path, "wb") as f:
            pickle.dump(texts, f)

        print(f"[✅] Saved FAISS index to: {index_path}")
        print(f"[✅] Saved chunk metadata to: {metadata_path}")
