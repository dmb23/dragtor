"""Functionality for splitting, embedding and storing loaded information"""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import chromadb.api

from dragtor.config import config


class RetrievalError(Exception):
    pass


class Index(ABC):
    def embed_chunks(self, chunks: list[str]) -> None:
        raise NotImplementedError

    def query(self, question: str, n_results: int) -> list[str]:
        raise NotImplementedError


@dataclass
class ChromaDBIndex(Index):
    """Use Chroma DB for embedding and vector manipulation"""

    _db_path: str = field(
        init=False, default=str(Path(config.base_path) / config.embeddings.chroma_path)
    )
    client: chromadb.api.ClientAPI = field(init=False)
    collection: chromadb.Collection = field(init=False)
    collection_name: str = "blog_1"

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=self._db_path)
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ValueError:
            self.collection = self.client.create_collection(self.collection_name)

    def embed_chunks(self, chunks: list[str]) -> None:
        ids = [str(hash(chunk)) for chunk in chunks]
        # TODO: this creates duplicate documents when run multiple times!
        self.collection.add(documents=chunks, ids=ids)

    def query(self, question: str, n_results: int = 5) -> list[str]:
        results = self.collection.query(query_texts=[question], n_results=n_results)
        try:
            documents = results["documents"][0]  # single str in list of queries
        except TypeError:
            raise RetrievalError("No documents were returned for query %s", question)
        return documents


class Chunker:
    """Split the knowledge base into chunks for retrieval"""

    def chunk_texts(self, texts: list[str]) -> list[str]:
        """Allow to select via config the strategy for splitting"""
        return self._chunk_by_line(texts)

    def _chunk_by_line(self, texts: list[str]) -> list[str]:
        chunks = [chunk for text in texts for chunk in text.split("\n\n")]
        return chunks
