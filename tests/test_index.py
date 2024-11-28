import numpy as np
import pytest

from dragtor.index.index import BasicIndex, LateChunkingIndex
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


@pytest.fixture
def late_chunking_index(empty_store):
    return LateChunkingIndex()


def test_late_chunking_init(late_chunking_index):
    assert late_chunking_index.tokenizer is not None
    assert late_chunking_index.embedding_model is not None
    assert late_chunking_index.chunker is not None
    assert late_chunking_index.reranker is not None


def test_map_chunks_to_tokens(late_chunking_index):
    # Create sample token offsets and chunk annotations
    token_offsets = np.array([[0, 5], [6, 10], [11, 15]])
    chunk_annotations = [(0, 10), (11, 15)]

    token_chunks = late_chunking_index._map_chunks_to_tokens(chunk_annotations, token_offsets)

    assert len(token_chunks) == len(chunk_annotations)
    assert token_chunks[0] == (0, 2)  # First chunk spans first two tokens
    assert token_chunks[1] == (2, 3)  # Second chunk is the last token


def test_late_chunking_index_texts(late_chunking_index, text):
    late_chunking_index.index_texts([text])
    collection = late_chunking_index.collection(late_chunking_index.chunk_collection_name)
    assert collection.count() > 0


def test_late_chunking_query(late_chunking_index, text):
    late_chunking_index.index_texts([text])
    results = late_chunking_index.query("test query")

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(doc, str) for doc in results)
