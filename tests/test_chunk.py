from dragtor.index.chunk import JinaTokenizerChunker, ParagraphChunker
import pytest


@pytest.mark.parametrize("ChunkingClass", [ParagraphChunker, JinaTokenizerChunker])
class TestChunking:
    def test_paragraph_chunks(self, ChunkingClass, example_text: str):
        c = ChunkingClass()
        chunks = c.chunk_texts([example_text])

        # eyeballed values here
        assert len(chunks) > 10
        for chunk in chunks:
            assert len(chunk) < 2000

    def test_paragraph_annotations(self, ChunkingClass, example_text: str):
        c = ChunkingClass()
        chunks, annotations = c.chunk_and_annotate(example_text)

        assert len(chunks) > 10
        assert len(annotations) > 10

        for c, a in zip(chunks, annotations):
            assert example_text[a[0] : a[1]] == c
