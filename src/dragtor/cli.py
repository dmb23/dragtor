"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger

from dragtor import data
from dragtor.config import config
from dragtor.index.index import get_index
from dragtor.llm import LocalDragtor


class Cli:
    """Ask the dRAGtor your questions about climbing injuries!

    Fully local RAG for knowledge from various sources about climbing injuries and treatment
    """

    @logger.catch
    def load(self):
        """Load remote data to local files for further processing"""
        urls = config.data.hoopers_urls
        logger.debug(f"loading the urls:\n{urls}")
        data.JinaLoader().load_jina_to_cache(urls)
        logger.info("Loaded data successfully")

    @logger.catch
    def index(self):
        """Create a Vector Store of embeddings of all loaded sources for retrieval"""
        loader = data.JinaLoader()
        full_texts = loader.get_cache()

        index = get_index()
        index.index_texts(full_texts)
        logger.info("Indexed all cached data successfully")

    @logger.catch
    def search(self, question: str) -> list[str]:
        """Find helpful information to answer the question"""
        index = get_index()
        logger.debug(f'Search content for: "{question}"')
        results = index.query(question)

        return results

    @logger.catch
    def ask(self, question: str) -> str:
        """Get an answer to your question based on the existing resources"""
        return LocalDragtor().answer(question)


def entrypoint():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
