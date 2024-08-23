from abc import ABC, abstractmethod

from loguru import logger

from dragtor.config import config
from dragtor.index import IndexStrategyError


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, docs: list[str], n_results: int) -> list[str]:
        pass


class DummyReranker(Reranker):
    def __init__(self) -> None:
        logger.debug("initializing Dummy Reranker")

    def rerank(self, question, docs: list[str], n_results: int = 0) -> list[str]:
        if n_results == 0:
            n_results = len(docs)
        return docs[:n_results]


class JinaReranker(Reranker):
    rerank = None

    def __init__(self) -> None:
        logger.debug("initializing Jina Reranker")


def get_reranker() -> Reranker:
    strat = config._select("reranker.strategy", default="default")
    match strat:
        case "default" | "dummy":
            return DummyReranker()
        case "jina":
            return JinaReranker()
        case _:
            raise IndexStrategyError(f"Unknown retrieval strategy {strat}")
