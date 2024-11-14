import pytest

from dragtor.index.chunk import JinaTokenizerChunker, ParagraphChunker, RecursiveCharacterChunker


@pytest.mark.parametrize(
    "ChunkingClass", [ParagraphChunker, JinaTokenizerChunker, RecursiveCharacterChunker]
)
class TestChunking:
    def test_creating_chunks(self, ChunkingClass, example_text: str):
        c = ChunkingClass()
        chunks = c.chunk_texts([example_text])

        # example text has ~2800 characters, super rough tests:
        assert len(chunks) > 2
        for chunk in chunks:
            assert len(chunk) < 1500

    def test_chunk_annotations(self, ChunkingClass, example_text: str):
        c = ChunkingClass()
        chunks, annotations = c.chunk_and_annotate(example_text)

        assert len(chunks) == len(annotations)

        for c, a in zip(chunks, annotations):
            assert example_text[a[0] : a[1]] == c
