from abc import ABC, abstractmethod

from loguru import logger
from sentence_transformers import CrossEncoder

from dragtor.config import config
from dragtor.index import IndexStrategyError


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, docs: list[str], n_results: int = 0) -> list[str]:
        pass


class DummyReranker(Reranker):
    def __init__(self) -> None:
        logger.debug("initializing Dummy Reranker")

    def rerank(self, question, docs: list[str], n_results: int = 0) -> list[str]:
        if n_results == 0:
            n_results = len(docs)
        return docs[:n_results]


class JinaReranker(Reranker):
    def __init__(self) -> None:
        self._model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)
        logger.debug("initializing Jina Reranker")

    def rerank(self, question: str, docs: list[str], n_results: int = 0) -> list[str]:
        if n_results == 0:
            n_results = len(docs)
        ranked = self._model.rank(question, docs, return_documents=True, top_k=n_results)
        return [r["text"] for r in ranked]


def get_reranker() -> Reranker:
    strat = config.select("reranker.strategy", default="default")
    match strat:
        case "default" | "dummy":
            return DummyReranker()
        case "jina":
            return JinaReranker()
        case _:
            raise IndexStrategyError(f"Unknown retrieval strategy {strat}")
