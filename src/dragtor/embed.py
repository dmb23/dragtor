"""Functionality for splitting, embedding and storing loaded information"""

from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import chromadb.api

from dragtor.config import config


@dataclass
class ChromaDBIndex:
    _db_path: str = field(
        init=False, default=str(Path(config.base_path) / config.embeddings.chroma_path)
    )
    client: chromadb.api.ClientAPI = field(init=False)
    collection_name: str = field(init=False, default="blog_1")

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=self._db_path)

    def embed_chunks(self, chunks: list[str]) -> None:
        try:
            collection = self.client.get_collection(self.collection_name)
        except ValueError:
            collection = self.client.create_collection(self.collection_name)

        ids = [str(hash(chunk)) for chunk in chunks]
        collection.add(documents=chunks, ids=ids)

    def chunk_texts(self, texts: list[str]) -> list[str]:
        chunks = [chunk for text in texts for chunk in text.split("\n\n")]
        return chunks
