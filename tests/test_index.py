import pytest

from dragtor.index.index import BasicIndex
from dragtor.index.rerank import get_reranker
from dragtor.index.store import BasicChromaStore


@pytest.fixture
def text() -> str:
    return "this is a long example text with lots of meaningless information. It includes multiple phrases.\n\nAnd multiple paragraphs, too!"


@pytest.fixture
def annotations() -> list[tuple[int, int]]:
    return [(0, 8), (8, 23)]


def test_basic_index_loading(text: str, empty_store: BasicChromaStore):
    i = BasicIndex(empty_store, get_reranker())
    chunks, _ = empty_store.chunker.chunk_and_annotate(text)

    i.index_texts([text])
    assert empty_store.collection.count() == len(chunks)


def test_basic_index_query(text: str, full_store):
    i = BasicIndex(full_store, get_reranker())
    i.index_texts([text])

    res = i.query("empty query prompt")
    assert len(res) > 0
    assert type(res) is list and type(res[0]) is str
