# app/ingestion/document_ingestor.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.parsers.dynamic_parser import parse_document
from app.ingestion.embedder import TextEmbedder
from app.ingestion.vectorstore import VectorStoreManager

class DocumentIngestor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedder = TextEmbedder()
        self.vectorstore_manager = VectorStoreManager()

    def ingest(self, file_path: str) -> str:
        try:
            print(f"[📄] Ingesting: {file_path}")
            raw_text = parse_document(file_path)
            if not raw_text:
                raise ValueError("Parsed document is empty.")

            print(f"[📖] Parsed text length: {len(raw_text)}")
            chunks = self.text_splitter.split_text(raw_text)
            print(f"[🔪] Split into {len(chunks)} chunks")

            embeddings = self.embedder.embed_texts(chunks)
            print(f"[🧠] Generated {len(embeddings)} embeddings")
            print(f"[DEBUG] First vector sample: {embeddings[0][:5] if embeddings else 'No vectors generated'}")

            doc_name = os.path.splitext(os.path.basename(file_path))[0]
            self.vectorstore_manager.save_vectorstore(embeddings, chunks, doc_name)

            print(f"[📦] Vectorstore saved for '{doc_name}'")
            return f"Document '{doc_name}' successfully ingested with {len(chunks)} chunks."
        except Exception as e:
            raise RuntimeError(f"Failed to ingest document '{file_path}': {str(e)}")
