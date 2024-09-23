from dragtor.index.index import LateChunkingIndex
import pytest
from pytest import fixture


@fixture
def text() -> str:
    return "this is a long example text with lots of meaningless information. It includes multiple phrases.\n\nAnd multiple paragraphs, too!"


@fixture
def annotations() -> list[tuple[int, int]]:
    return [(0, 8), (8, 23)]


@pytest.mark.skip()
def test_late_embeddings(text):
    i = LateChunkingIndex.from_defaults()

    chunks, _ = i.chunker.chunk_and_annotate(text)
    chunk_embeddings = i._calculate_late_embeddings_v2(text)

    assert len(chunk_embeddings) == len(chunks)
    for emb in chunk_embeddings:
        assert emb.shape[0] == 1
