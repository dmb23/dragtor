from dataclasses import dataclass

from loguru import logger

from dragtor.config import config
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import VectorStore, get_store


@dataclass
class Index:
    chunker: Chunker
    store: VectorStore
    reranker: Reranker

    def index_texts(self, texts: list[str]) -> None:
        chunks = self.chunker.chunk_texts(texts)
        self.store.add_chunks(chunks)

    def query(self, question: str) -> list[str]:
        n_results = config._select("store.n_query", default=5)
        n_ranked = config._select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        logger.debug(
            f"extracting {n_results} documents from the store, then rerank down to {n_ranked}"
        )
        found_docs = self.store.query(question, n_results)
        logger.debug(f"obtained {len(found_docs)} documents")
        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"obtained {len(ranked_docs)} after reranking")
            return ranked_docs
        return found_docs


def get_index() -> Index:
    chunker = get_chunker()
    store = get_store()
    reranker = get_reranker()
    logger.debug("initialized all parts of index")
    return Index(chunker, store, reranker)
