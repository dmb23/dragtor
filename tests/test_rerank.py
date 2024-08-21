from dragtor.index import rerank
import pytest


@pytest.fixture
def docs() -> list[str]:
    return [
        "Bananas are yellow",
        "I have eaten an apple",
        "The red car is fast",
    ]


@pytest.fixture
def question() -> str:
    return "What is the speed?"


def test_dummy_rerank(docs: list[str], question: str):
    r = rerank.DummyReranker()
    ranked_docs = r.rerank(question, docs)

    for old, new in zip(docs, ranked_docs):
        assert old == new

    ranked_docs = r.rerank(question, docs, n_results=2)
    assert len(ranked_docs) == 2


def test_jina_rerank(docs: list[str], question: str):
    r = rerank.JinaReranker()

    ranked_docs = r.rerank(question, docs)
    assert ranked_docs[0] == "The red car is fast"

    ranked_docs = r.rerank(question, docs, n_results=2)
    assert len(ranked_docs) == 2
