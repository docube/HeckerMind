# app/ingestion/embedder.py

import os
from typing import List
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

class TextEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the embedder with a specific model.
        """
        self.model_name = model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY is missing. Please set it in your .env file or environment variables."
            )

        self.embedder = OpenAIEmbeddings(
            model=self.model_name,
            api_key=SecretStr(api_key)
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        print(f"[CALL] Embedding {len(texts)} texts using model {self.model_name}")
        try:
            embeddings = self.embedder.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"[ERROR] Failed to embed: {e}")
            return []
