from abc import ABC
from dataclasses import dataclass, field

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Document, Embedding
from chromadb.utils import embedding_functions
from loguru import logger

from dragtor import config
from dragtor.index import IndexStrategyError


@dataclass
class Embedder(ABC):
    ef: EmbeddingFunction[Documents] = field(init=False)

    def embed_query(self, query: Document) -> Embedding:
        return self.ef([query])[0]

    def embed_documents(self, documents: Documents) -> Embeddings:
        return self.ef(documents)


@dataclass
class DefaultEmbedder(Embedder):
    """Use the ChromaDB default embeddings (all-MiniLM-L6-v2 Sentence-Transformers)"""

    ef: EmbeddingFunction[Documents] = field(
        init=False, default_factory=embedding_functions.DefaultEmbeddingFunction
    )


@dataclass
class JinaEmbedder(Embedder):
    """Embed using the base jina-ai embedding model

    (could be switched to a faster one via jina-embeddings-v2-small-en"""

    _model_path: str = "jinaai/jina-embeddings-v2-base-en"

    def __post_init__(self):
        # trust_remote_code is needed to use the encode method
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        max_length = config.conf.select("embeddings.jina.max_seq_length", default=1024)
        self.ef._model.max_seq_length = max_length
        logger.debug(f"initialized jina embeddings with maximum seq length {max_length}")


def get_embedder() -> Embedder:
    """get the embedder according to config"""
    strat = config.conf.select("embeddings.strategy", default="default")
    match strat:
        case "default" | "chromadb":
            return DefaultEmbedder()
        case "jina":
            return JinaEmbedder()
        case _:
            raise IndexStrategyError(f"unknown embedding strategy {strat}")
