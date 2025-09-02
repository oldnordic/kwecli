"""
RAG (Retrieval-Augmented Generation) facade using LlamaIndex.
Provides unified methods: ingest, search, answer.
"""
import os
from typing import List, Union

# LlamaIndex v0.10+ imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.query_engine import RetrieverQueryEngine


class RAG:
    """Facade for LlamaIndex RAG with in-memory and persistent storage."""

    def __init__(self, index_dir: str = 'index'):
        self.index_dir = index_dir
        self.index = None
        os.makedirs(self.index_dir, exist_ok=True)

    def ingest(self, paths: Union[str, List[str]]) -> None:
        """
        Ingest documents from file paths or directories.
        paths: single path or list of paths (files or directories)
        """
        if isinstance(paths, str):
            paths = [paths]
        
        docs = []
        for path in paths:
            if os.path.isdir(path):
                reader = SimpleDirectoryReader(path)
                docs.extend(reader.load_data())
            elif os.path.isfile(path):
                reader = SimpleDirectoryReader(input_files=[path])
                docs.extend(reader.load_data())
        
        self.index = VectorStoreIndex.from_documents(docs)
        self.index.storage_context.persist(persist_dir=self.index_dir)

    def _load_index(self):
        if not self.index:
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
                self.index = VectorStoreIndex.from_storage(storage_context)
            except Exception:
                self.index = VectorStoreIndex.from_documents([])

    def search(self, query: str, top_k: int = 4) -> dict:
        """
        Search the index for relevant documents.
        """
        self._load_index()
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        return {
            'response': str(response),
            'source_nodes': [str(node.text[:200]) for node in response.source_nodes] if hasattr(response, 'source_nodes') else []
        }

    def answer(self, question: str, top_k: int = 4) -> dict:
        """
        Answer a question using RAG: search + LLM answer.
        """
        self._load_index()
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)
        return {
            'answer': str(response),
            'sources': [str(node.text[:200]) for node in response.source_nodes] if hasattr(response, 'source_nodes') else []
        }

# Singleton instance
rag = RAG()