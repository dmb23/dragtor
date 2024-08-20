from dragtor.config import config
from dragtor.index import IndexStrategyError


class Chunker:
    """Split the knowledge base into chunks for retrieval"""

    def chunk_texts(self, texts: list[str]) -> list[str]:
        """Allow to select via config the strategy for splitting"""
        chunk_conf = config.get("chunking", {})
        match chunk_conf.get("strategy", "default"):
            case "jina_lines" | "default":
                return self._chunk_by_paragraph(texts)
            case _:
                raise IndexStrategyError(f"unknown chunking strategy {chunk_conf.strategy}")

    def _chunk_by_paragraph(self, texts: list[str]) -> list[str]:
        """Chunk the parsed output of Jina Reader by double-newlines"""
        chunks = [chunk for text in texts for chunk in text.split("\n\n")]
        return chunks
