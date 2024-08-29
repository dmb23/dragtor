from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import chromadb
import chromadb.api
from loguru import logger

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


def _build_collection_name() -> str:
    chunk_strat = config._select("chunking.strategy", default="jina_line")
    embed_strat = config._select("embeddings.strategy", default="chromadb")
    rerank_strat = config._select("embeddings.strategy", default="no_rerank")

    return "__".join([chunk_strat, embed_strat, rerank_strat])


@dataclass
class ChromaDBStore(VectorStore):
    _db_path: str = str(Path(config.base_path) / config.store.chromadb.path)
    client: chromadb.api.ClientAPI = field(init=False)
    collection: chromadb.Collection = field(init=False)

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=self._db_path)

        collection_name = config._select("store.chromadb.collection_name", default=None)
        if collection_name is None:
            collection_name = _build_collection_name()

        distance = config._select("store.chromadb.distance", default="l2")
        logger.debug(f"Using Collection '{collection_name}' with distance '{distance}'")
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=self.embedder.ef,  # pyright: ignore [ reportArgumentType ]
            metadata={
                "hnsw:space": distance,
            },
        )
        logger.debug(f"Collection contains initially {len(self.collection.get()['ids'])} items.")

    def add_chunks(self, chunks: list[str]) -> None:
        n_init = len(self.collection.get()["ids"])
        chunks = list(set(chunks))
        ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
        embeddings = self.embedder.ef(chunks)
        self.collection.add(documents=chunks, ids=ids, embeddings=embeddings)
        n_post = len(self.collection.get()["ids"])
        logger.debug(
            f"Tried to add {len(chunks)} new elements to collection, size increased from {n_init} to {n_post}"
        )

    def query(self, question: str, n_results: int = 5) -> list[str]:
        embedding = self.embedder.embed_query(question)
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        try:
            # only a single embedding (of a single question) is queried
            documents = results["documents"][0]  # pyright: ignore [ reportOptionalSubscript ]
            logger.debug(f"collected {len(documents)} results from ChromaDB")
        except TypeError:
            raise RetrievalError("No documents were returned for query %s", question)
        return documents


def get_store() -> ChromaDBStore:
    match strat := config._select("store.strategy", default="default"):
        case "chromadb" | "default":
            embedder = get_embedder()
            return ChromaDBStore(embedder)
        case _:
            raise RetrievalError(f"Unknown strategy for Vector Store: {strat}")
