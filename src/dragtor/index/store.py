from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import chromadb.api
from loguru import logger

from dragtor import config
from dragtor.data.data import Document
from dragtor.index import RetrievalError
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.embed import Embedder, get_embedder
from dragtor.utils import ident


@dataclass
class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def query(self, question: str, n_results: int) -> list[str]:
        pass


def _build_collection_name() -> str:
    chunk_strat = config.conf.select("chunking.strategy", default="jina_line")
    embed_strat = config.conf.select("embeddings.strategy", default="chromadb")
    rerank_strat = config.conf.select("reranker.strategy", default="no_rerank")

    return "__".join([chunk_strat, embed_strat, rerank_strat])


def _get_default_db_directory() -> str:
    return str(Path(config.conf.base_path) / config.conf.store.chromadb.path)


@dataclass
class BasicChromaStore(VectorStore):
    chunker: Chunker = field(default_factory=get_chunker)
    embedder: Embedder = field(default_factory=get_embedder)
    _db_path: str = field(default_factory=_get_default_db_directory)
    client: chromadb.api.ClientAPI = field(init=False)
    collection: chromadb.Collection = field(init=False)

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=self._db_path)

        collection_name = config.conf.select("store.chromadb.collection_name", default=None)
        if collection_name is None:
            collection_name = _build_collection_name()

        distance = config.conf.select("store.chromadb.distance", default="l2")
        logger.debug(f"Using Collection '{collection_name}' with distance '{distance}'")
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=self.embedder.ef,  # pyright: ignore [ reportArgumentType ]
            metadata={
                "hnsw:space": distance,
            },
        )
        logger.debug(f"Collection contains initially {len(self.collection.get()['ids'])} items.")

    def add_documents(self, documents: list[Document]) -> None:
        n_init = self.collection.count()
        
        # Process each document separately to maintain metadata connection
        all_chunks = []
        all_ids = []
        all_metadata = []
        
        for doc in documents:
            # Chunk the document
            chunks = self.chunker.chunk_texts([doc.content])
            
            # Create metadata for each chunk
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(ident(chunk))
                
                # Combine document metadata with chunk-specific info
                metadata = {
                    "title": doc.title,
                    "id": doc.id,
                    "author": doc.author if doc.author else "",
                }
                if doc.metadata:
                    metadata.update(doc.metadata)
                all_metadata.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedder.ef(all_chunks)
        
        # Add to collection with metadata
        self.collection.add(
            documents=all_chunks,
            ids=all_ids,
            embeddings=embeddings,
            metadatas=all_metadata
        )
        n_post = self.collection.count()
        
        logger.debug(
            f"Tried to add {len(chunks)} new elements to collection, "
            f"size increased from {n_init} to {n_post}"
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


def get_store() -> VectorStore:
    match strat := config.conf.select("store.strategy", default="default"):
        case "chromadb" | "default":
            return BasicChromaStore()
        case _:
            raise RetrievalError(f"Unknown strategy for Vector Store: {strat}")
