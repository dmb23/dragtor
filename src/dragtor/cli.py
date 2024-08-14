"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger

from dragtor import data, embed
from dragtor.config import config


class Cli:
    def load(self):
        urls = config.data.hoopers_urls
        logger.debug(f"loading the urls:\n{urls}")
        data.JinaLoader().load_jina_to_cache(urls)
        logger.info("Loaded data successfully")

    def index(self):
        loader = data.JinaLoader()
        full_texts = loader.get_cache()
        chunks = embed.Chunker().chunk_texts(full_texts)

        index = embed.ChromaDBIndex()
        index.embed_chunks(chunks)
        logger.info("Indexed all cached data successfully")

    def search(self, question: str, n_results=5):
        logger.debug(f'Search content for: "{question}"')
        index = embed.ChromaDBIndex()
        results = index.query(question, n_results)
        logger.info("Found information: {results}".format(results="\n" + "\n".join(results)))

    def ask(self):
        logger.info("ask - not implemented")


def entrypoint():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
