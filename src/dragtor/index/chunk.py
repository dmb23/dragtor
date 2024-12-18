from abc import ABC, abstractmethod

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

from dragtor import config
from dragtor.index.embed import JinaEmbedder


class Chunker(ABC):
    def chunk_texts(self, texts: list[str]) -> list[str]:
        """Collect only the chunks for multiple texts"""
        all_chunks = [chunk for text in texts for chunk in self.chunk_and_annotate(text)[0]]
        return all_chunks

    @abstractmethod
    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Split a single text into chunks.

        Returns:
            chunks: list of strings
            annotations: list of tuples of (start,end) position for each chunk
        """
        pass

    def _find_chunks(self, text: str, chunks: list[str]) -> list[tuple[int, int]]:
        """Find the position of chunks in a given text.

        Raises: ValueError if any chunk is not present in the text
        """
        i_latest = 0
        annotations: list[tuple[int, int]] = []
        for chunk in chunks:
            i_start: int = text.find(chunk, i_latest)
            i_latest = i_start + len(chunk)
            annotation = (i_start, i_latest)
            annotations.append(annotation)
        return annotations


class ParagraphChunker(Chunker):
    r"""Split texts by markdown paragraphs `\n\n`"""

    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        chunks = text.split("\n\n")
        chunk_annotations = self._find_chunks(text, chunks)

        return chunks, chunk_annotations


class JinaTokenizerChunker(Chunker):
    """Use JINA tokenizer API for chunking"""

    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        max_chunk_length = config.conf.select("chunking.jina_tokenizer.max_chunk_length", 1000)
        # Define the API endpoint and payload
        url = "https://tokenize.jina.ai/"
        payload = {"content": text, "return_chunks": "true", "max_chunk_length": max_chunk_length}

        # Make the API request
        response = requests.post(url, json=payload)
        response_data = response.json()

        # Extract chunks and positions from the response
        chunks = response_data.get("chunks", [])
        chunk_positions = response_data.get("chunk_positions", [])

        # Adjust chunk positions to match the input format
        span_annotations = [(start, end) for start, end in chunk_positions]

        return chunks, span_annotations


class RecursiveCharacterChunker(Chunker):
    """Use Langchain API for chunking"""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            # Follow config's max chunk length
            chunk_size=config.conf.select("chunking.recursive_character.max_chunk_length", 1000),
            chunk_overlap=config.conf.select("chunking.recursive_character.chunk_overlap", 50),
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        chunks = []
        annotations = []

        # Start the count for annotation
        start_position = 0
        for chunk in self.splitter.split_text(text):
            # need to search for the chunk, overlap can be variable
            chunk_start = text.find(chunk, start_position)
            chunk_end = chunk_start + len(chunk)
            assert text[chunk_start:chunk_end] == chunk

            # Append the list for all chunks & annotation
            chunks.append(chunk)
            annotations.append((chunk_start, chunk_end))

            # Update start_position value to end position, with excluding overlap
            start_position = chunk_end - self.splitter._chunk_overlap

        return chunks, annotations


class LCSemanticChunker(Chunker):
    """Use Langchain Semantic Chunking"""

    def __init__(self) -> None:
        self.embedder = JinaEmbedder()
        self.splitter = SemanticChunker(
            self.embedder,
            breakpoint_threshold_type="percentile",
        )

    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        chunks = []
        annotations = []

        # The sentence splitting regex in the splitter can change whitespace in the chunks
        chunk_start = 0
        chunk_end = 0
        for chunk in self.splitter.split_text(text):
            for char in chunk:
                if char in " \t\n":
                    continue
                chunk_end = text.find(char, chunk_end) + 1

            chunks.append(text[chunk_start:chunk_end])
            annotations.append((chunk_start, chunk_end))

            chunk_start = chunk_end

        return chunks, annotations


def get_chunker() -> Chunker:
    match config.conf.select("chunking.strategy", "default"):
        case "default" | "paragraph":
            return ParagraphChunker()
        case "jina_tokenizer":
            return JinaTokenizerChunker()
        case "recursive_character":
            return RecursiveCharacterChunker()
        case "semantic_segmentation":
            return LCSemanticChunker()
        case _:
            raise config.ConfigurationError(
                f"invalid strategy to select chunker: {config.conf.select('chunking.strategy')}"
            )
