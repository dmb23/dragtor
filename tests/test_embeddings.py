"""smoke tests to run embedding functions"""

from dragtor.index import embed
import pytest


@pytest.fixture
def documents() -> list[str]:
    return [
        "this is a document",
        "this is an other document",
    ]


@pytest.fixture
def query() -> str:
    return "where could I find a document?"


def test_default_embedder(documents: list[str], query: str):
    e = embed.DefaultEmbedder()

    doc_embs = e.embed_documents(documents)
    query_embs = e.embed_query(query)

    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == len(query_embs)


def test_jina_embedder(documents: list[str], query: str):
    e = embed.JinaEmbedder()

    doc_embs = e.embed_documents(documents)
    query_embs = e.embed_query(query)

    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == len(query_embs)
