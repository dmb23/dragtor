"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger

from dragtor import data, embed
from dragtor.config import config
from dragtor.llm import Generator


class Cli:
    """Ask the dRAGtor your questions about climbing injuries!

    Fully local RAG for knowledge from various sources about climbing injuries and treatment
    """

    def load(self):
        """Load remote data to local files for further processing"""
        urls = config.data.hoopers_urls
        logger.debug(f"loading the urls:\n{urls}")
        data.JinaLoader().load_jina_to_cache(urls)
        logger.info("Loaded data successfully")

    def index(self):
        """Create a Vector Store of embeddings of all loaded sources for retrieval"""
        loader = data.JinaLoader()
        full_texts = loader.get_cache()
        chunks = embed.Chunker().chunk_texts(full_texts)

        index = embed.ChromaDBIndex()
        index.embed_chunks(chunks)
        logger.info("Indexed all cached data successfully")

    def search(self, question: str, n_results=5) -> list[str]:
        """Find helpful information to answer the question"""
        logger.debug(f'Search content for: "{question}"')
        index = embed.ChromaDBIndex()
        results = index.query(question, n_results)

        return results

    def ask(self, question: str) -> str:
        """Get an answer to your question based on the existing resources"""
        return Generator().query(question)


def entrypoint():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
