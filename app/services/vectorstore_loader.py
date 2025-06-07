# app/services/vectorstore_loader.py

import os
import faiss
import pickle
from typing import List
import numpy as np
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from app.config.settings import get_settings

settings = get_settings()

VECTORSTORE_DIR = "vectorstore"

def list_all_documents() -> List[str]:
    """List all document folders inside the vectorstore directory."""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    return [
        name for name in os.listdir(VECTORSTORE_DIR)
        if os.path.isdir(os.path.join(VECTORSTORE_DIR, name))
    ]

def load_vectorstore(document_name: str):
    """Manually load a FAISS index and associated metadata (not using langchain_community FAISS wrapper)."""
    doc_path = os.path.join(VECTORSTORE_DIR, document_name)
    index_file = os.path.join(doc_path, "index.faiss")
    metadata_file = os.path.join(doc_path, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        return None

    try:
        index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            texts = pickle.load(f)

        return index, texts
    except Exception as e:
        print(f"Failed to load vectorstore for {document_name}: {e}")
        return None

def get_relevant_documents(query: str, k_per_doc: int = 3) -> List[Document]:
    """Search across all vectorstores and return top-k relevant chunks per document."""
    embedder = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    query_embedding = embedder.embed_query(query)

    results = []
    documents = list_all_documents()

    for doc_name in documents:
        loaded = load_vectorstore(doc_name)
        if loaded is None:
            continue

        index, texts = loaded
        try:
            D, I = index.search(np.array([query_embedding]).astype("float32"), k_per_doc)
            for i in I[0]:
                if i != -1 and i < len(texts):
                    results.append(Document(page_content=texts[i], metadata={"source": doc_name}))
        except Exception as e:
            print(f"Error searching in vectorstore '{doc_name}': {e}")
            continue

    return results
