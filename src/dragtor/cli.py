"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger

from dragtor import data
from dragtor.config import config


class Cli:
    def load(self):
        urls = config.data.hoopers_urls
        logger.debug(f"loading the urls:\n{urls}")
        data.JinaLoader().load_jina_to_cache(urls)
        logger.info("Loaded data successfully")

    def index(self):
        logger.info("index - not implemented")

    def search(self):
        logger.info("search - not implemented")

    def ask(self):
        logger.info("ask - not implemented")


def entry():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
