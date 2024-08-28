from abc import ABC, abstractmethod

import requests

from dragtor.config import ConfigurationError, config


class Chunker(ABC):
    def chunk_texts(self, texts: list[str]) -> list[str]:
        """Collect only the chunks for multiple texts"""
        all_chunks = [chunk for text in texts for chunk in self.chunk_and_annotate(text)[0]]
        return all_chunks

    @abstractmethod
    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        pass

    def _find_chunks(self, text: str, chunks: list[str]) -> list[tuple[int, int]]:
        """Find the position of chunks in a given text.

        Raises: ValueError if any chunk is not present in the text
        """
        i_latest = 0
        annotations: list[tuple[int, int]] = []
        for chunk in chunks:
            i_start: int = text.find(chunk, start=i_latest)
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
    def chunk_and_annotate(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        max_chunk_length = config.select("chunker.jina_tokenizer.max_chunk_length", 1000)
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


def get_chunker() -> Chunker:
    match config.select("chunking.strategy", "default"):
        case "default" | "paragraph":
            return ParagraphChunker()
        case "jina_tokenizer":
            return JinaTokenizerChunker()
        case _:
            raise ConfigurationError(
                f"invalid strategy to select chunker: {config.select('chunking.strategy')}"
            )
