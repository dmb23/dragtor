"""An 'Index' is a class that manages all interaction with possible context information.

Main entrypoints:
- index_texts: add information from new texts to the index
- query: obtain relevant information for a prompt / question

I want to keep indexing very flexible. Funcitonality can vary:
- add parts of the original texts verbatim and obtain the most relevant ones
- employ more complicated embedding schemes, which require more information than just the text
- create additional information (summaries, possible questions) to store on top of the texts
- obtain the full document that contains a matching snippet
- ...

NOTE: an Index could contain in the future agentic steps after retrieval.
At that point it provides an interesting additional layer of abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from loguru import logger

from dragtor import config
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import VectorStore, get_store


@dataclass
class Index(ABC):
    @abstractmethod
    def index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        pass

    @abstractmethod
    def query(self, question: str) -> list[str]:
        pass


@dataclass
class BasicIndex(Index):
    store: VectorStore
    reranker: Reranker

    def index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self.store.add_documents(texts, metadata)

    def query(self, question: str) -> list[str]:
        n_results = config.conf.select("store.n_query", default=5)
        n_ranked = config.conf.select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        found_docs = self.store.query(question, n_results)
        logger.debug(f"extracted {len(found_docs)} documents")
        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"reranked down to {len(ranked_docs)} documents")
            return ranked_docs
        return found_docs

    @classmethod
    def from_defaults(cls) -> Self:
        store = get_store()
        reranker = get_reranker()
        logger.debug("initialized all parts of Index")
        return cls(store, reranker)


def get_index() -> Index:
    strat = config.conf.select("index.strategy", "default")
    match strat:
        case "default" | "basic":
            return BasicIndex.from_defaults()
        case _:
            raise ValueError(f"Uknown value '{strat}' for index.strategy!")
