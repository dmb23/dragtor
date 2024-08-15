from abc import ABC, abstractmethod

from dragtor.config import config
from dragtor.index import IndexStrategyError


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, docs: list[str], n_results: int) -> list[str]:
        pass


class DummyReranker(Reranker):
    def rerank(self, question, docs: list[str], n_results: int) -> list[str]:
        return docs[:n_results]


def get_reranker() -> Reranker:
    strat = config.get("reranker.strategy", "default")
    match strat:
        case "default" | "dummy":
            return DummyReranker()
        case _:
            raise IndexStrategyError(f"Unknown retrieval strategy {strat}")
