import math
import re

from dragtor.index.chunk import JinaTokenizerChunker, ParagraphChunker, RecursiveCharacterChunker
import pytest
from dragtor import config

from tests.conftest import example_text

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

# Helper for Langchain recursive character chunker tests
def clean_text(text: str) -> str:
    """Remove unwanted characters from the text for comparison. Because langchain automatically remove start & end whitespaces of each chunk."""
    return re.sub(r"[^\w.,'?\-]", "", text)

def calculate_expected_chunk_count(text_length: int, chunk_size: int) -> int:
    """Calculate the expected number of chunks based on the text length and chunk size."""
    return math.ceil(text_length / chunk_size)

def test_recursive_chunk(example_text: str):
    chunk_size = config.conf.chunking.langchain_tokenizer.max_chunk_length
    clean_example_text = clean_text(example_text)

    lang = RecursiveCharacterChunker()
    chunks = lang.chunk_texts([example_text])

    expected_chunk_count = calculate_expected_chunk_count(len(example_text), chunk_size)

    assert len(chunks) == expected_chunk_count
    for chunk in chunks:
        assert len(chunk) < 1000

    reconstructed_text = "".join(clean_text(chunk) for chunk in chunks)
    assert reconstructed_text == clean_example_text

def test_recursive_annotation(example_text: str):
    chunk_size = config.conf.chunking.langchain_tokenizer.max_chunk_length

    lang = RecursiveCharacterChunker()
    chunks, annotations = lang.chunk_and_annotate(example_text)

    expected_chunk_count = calculate_expected_chunk_count(len(example_text), chunk_size)
    assert len(annotations) == expected_chunk_count

    chunk_overlap = config.conf.chunking.langchain_tokenizer.chunk_overlap
    for i in range(1, len(annotations)):
        prev_end = annotations[i - 1][1]
        curr_start = annotations[i][0]
        assert curr_start == prev_end - chunk_overlap