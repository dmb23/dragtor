from abc import ABC, abstractmethod

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dragtor import config


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
    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        chunks = []
        annotations = []

        recursive_text_splitter = RecursiveCharacterTextSplitter(
            # Follow config's max chunk length
            chunk_size=config.conf.select("chunking.langchain_tokenizer.max_chunk_length", 1000),
            chunk_overlap=config.conf.select("chunking.langchain_tokenizer.chunk_overlap", 0),
            length_function=len,
            is_separator_regex=False,
        )

        # Start the count for annotation
        start_position = 0
        for chunk in recursive_text_splitter.split_text(text):
            # End annotation for each chunk
            end_position = start_position + len(chunk)

            # Append the list for all chunks & annotation
            chunks.append(chunk)
            annotations.append(
                (start_position, end_position)
            )

            # Update start_position value to end position, with excluding overlap
            start_position = end_position - recursive_text_splitter._chunk_overlap

        return chunks, annotations


def get_chunker() -> Chunker:
    match config.conf.select("chunking.strategy", "default"):
        case "default" | "paragraph":
            return ParagraphChunker()
        case "jina_tokenizer":
            return JinaTokenizerChunker()
        case "langchain":
            return RecursiveCharacterChunker()
        case _:
            raise config.ConfigurationError(
                f"invalid strategy to select chunker: {config.conf.select('chunking.strategy')}"
            )
