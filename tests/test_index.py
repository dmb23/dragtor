from dragtor.index.index import BasicIndex, LateChunkingIndex
import pytest


@pytest.fixture
def text() -> str:
    return "this is a long example text with lots of meaningless information. It includes multiple phrases.\n\nAnd multiple paragraphs, too!"


@pytest.fixture
def annotations() -> list[tuple[int, int]]:
    return [(0, 8), (8, 23)]


# @pytest.mark.parametrize("IndexClass", [BasicIndex, LateChunkingIndex])
# TODO: fix LateChunkingIndex
@pytest.mark.parametrize("IndexClass", [BasicIndex])
class TestIndex:
    def test_index_loading(self, IndexClass, text: str, empty_store):
        i = IndexClass.from_defaults()
        chunks, _ = i.chunker.chunk_and_annotate(text)

        i.index_texts([text])
        assert i.store.collection.count() == len(chunks)

    def test_index_query(self, IndexClass, text: str, full_store):
        i = IndexClass.from_defaults()
        i.index_texts([text])

        res = i.query("empty query prompt")
        assert len(res) > 0
        assert type(res) is not str


@pytest.mark.skip(reason="Late Chunking is not finished")
def test_late_embeddings(text):
    i = LateChunkingIndex.from_defaults()

    chunks, _ = i.chunker.chunk_and_annotate(text)
    chunk_embeddings = i._calculate_late_embeddings_v2(text)

    assert len(chunk_embeddings) == len(chunks)
    for emb in chunk_embeddings:
        assert emb.shape[0] == 1
