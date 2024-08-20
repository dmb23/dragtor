from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import chromadb
import chromadb.api

from dragtor.config import config
from dragtor.index import RetrievalError
from dragtor.index.embed import Embedder, get_embedder


@dataclass
class VectorStore(ABC):
    embedder: Embedder

    @abstractmethod
    def add_chunks(self, chunks: list[str]) -> None:
        pass

    @abstractmethod
    def query(self, question: str, n_results: int) -> list[str]:
        pass


@dataclass
class ChromaDBStore(VectorStore):
    _db_path: str = str(Path(config.base_path) / config.store.chromadb.path)
    client: chromadb.api.ClientAPI = field(init=False)
    collection: chromadb.Collection = field(init=False)

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=self._db_path)

        collection_name = config._select(
            "store.chromadb.collection_name", default="default_collection"
        )
        if config._select("store.chromadb.reset_on_load", default=False):
            try:
                self.client.delete_collection(collection_name)
            except ValueError:
                pass

        distance = config._select("store.chromadb.distance", default="l2")
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=self.embedder.ef,  # pyright: ignore [ reportArgumentType ]
            metadata={
                "hnsw:space": distance,
            },
        )

    def add_chunks(self, chunks: list[str]) -> None:
        ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
        embeddings = self.embedder.ef(chunks)
        self.collection.add(documents=chunks, ids=ids, embeddings=embeddings)

    def query(self, question: str, n_results: int = 5) -> list[str]:
        embedding = self.embedder.embed_query(question)
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        try:
            # only a single document is queried
            documents = results["documents"][0]  # pyright: ignore [ reportOptionalSubscript ]
        except TypeError:
            raise RetrievalError("No documents were returned for query %s", question)
        return documents


def get_store() -> VectorStore:
    match strat := config._select("store.strategy", default="default"):
        case "chromadb" | "default":
            embedder = get_embedder()
            return ChromaDBStore(embedder)
        case _:
            raise RetrievalError(f"Unknown strategy for Vector Store: {strat}")
