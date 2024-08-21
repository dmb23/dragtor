from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from dragtor.config import config
from dragtor.index import IndexStrategyError


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, docs: list[str], n_results: int) -> list[str]:
        pass


class DummyReranker(Reranker):
    def rerank(self, question, docs: list[str], n_results: int = 0) -> list[str]:
        if n_results == 0:
            n_results = len(docs)
        return docs[:n_results]


class JinaReranker(Reranker):
    def __init__(self) -> None:
        self._model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)

    def rerank(self, question: str, docs: list[str], n_results: int = 0) -> list[str]:
        if n_results == 0:
            n_results = len(docs)
        results = self._model.rank(question, docs, return_documents=True, top_k=n_results)

        return [res["text"] for res in results]


def get_reranker() -> Reranker:
    strat = config._select("reranker.strategy", default="default")
    match strat:
        case "default" | "dummy":
            return DummyReranker()
        case "jina":
            return JinaReranker()
        case _:
            raise IndexStrategyError(f"Unknown retrieval strategy {strat}")
