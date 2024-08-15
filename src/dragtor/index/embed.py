from abc import ABC
from dataclasses import dataclass, field

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Document, Embedding
from chromadb.utils import embedding_functions

from dragtor.config import config
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


class MxbEmbedder(Embedder):
    """Embed using a mixedbread-ai embedding model"""

    _model_path: str = "mixedbread-ai/mxbai-embed-large-v1"

    def __post_init__(self):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore [reportAttributeAccessIssue]
            model_name=self._model_path,
            truncate_dim=1024,
        )

    def embed_query(self, query: Document) -> Embedding:
        """Specific format required to embed queries for MixedbreadAI"""
        adj_query = f"Represent this sentence for searching relevant passages: {query}"

        return self.ef([adj_query])[0]


def get_embedder() -> Embedder:
    """get the embedder according to config"""
    strat = config.get("embeddings.strategy", "default")
    match strat:
        case "default" | "chromadb":
            return DefaultEmbedder()
        case "mixedbread":
            return MxbEmbedder()
        case _:
            raise IndexStrategyError(f"unknown embedding strategy {strat}")
