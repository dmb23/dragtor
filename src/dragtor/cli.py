"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger
from omegaconf import OmegaConf

from dragtor import config, data
from dragtor.index.index import get_index
from dragtor.index.store import ChromaDBStore
from dragtor.llm import LocalDragtor
from dragtor.llm.eval import QuestionEvaluator
from dragtor.utils import ident


class Cli:
    """Ask the dRAGtor your questions about climbing injuries!

    Fully local RAG for knowledge from various sources about climbing injuries and treatment
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            logger.debug(f"updateing {k} with value {v}")
            OmegaConf.update(config.conf, k, v)

    @logger.catch
    def load(self):
        """Load remote data to local files for further processing"""
        urls = config.conf.data.hoopers_urls
        logger.debug(f"loading the urls:\n{urls}")
        data.JinaLoader().load_jina_to_cache(urls)
        logger.info("Loaded data successfully")

    def clear_index(self):
        """Reset all data already in the index"""
        index = get_index()
        if type(index.store) is not ChromaDBStore:
            raise ValueError("clear_index only works for a Chroma DB Store!")

        for c in index.store.client.list_collections():
            index.store.client.delete_collection(c.name)

    @logger.catch
    def index(self):
        """Create a Vector Store of embeddings of all loaded sources for retrieval"""
        loader = data.JinaLoader()
        full_texts = loader.get_cache()

        index = get_index()
        index.index_texts(full_texts)
        logger.info("Indexed all cached data successfully")

    @logger.catch
    def preload(self):
        """Create pre-loaded state files for all loaded sources for retrieval"""
        loader = data.JinaLoader()
        full_texts = loader.get_cache()

        ld = LocalDragtor()
        for i, text in enumerate(full_texts):
            text_id = ident(text)
            messages = ld._to_messages(question="", context=text)
            filename = f"{text_id}.bin"
            ld.llm.store_state(messages, filename)
            logger.info(f"Preloaded {i+1}/{len(full_texts)} texts")

    @logger.catch
    def search(self, question: str) -> str:
        """Find helpful information to answer the question"""
        index = get_index()
        logger.debug(f'Search content for: "{question}"')
        results = index.query(question)

        return "\n---\n".join([f"{i+1}: {r}" for i, r in enumerate(results)])

    @logger.catch
    def ask(self, question: str, statefile: str = "") -> str:
        """Get an answer to your question based on the content of a file from the index cache"""
        return LocalDragtor().chat(question, statefile)

    @logger.catch
    def transform(self):
        """Create additional features from available sources.

        - extract topics per source
        - create summaries per topic
        - create question / answer pairs for each topic
        """
        raise NotImplementedError

    def eval(self, question: str):
        """Evaluate the performance of the configured RAG setup.

        - evaluate how many of the propositions in a given answer are based on the sources.
        - possibly: evaluate answers to reference questions against gold truths
        """
        evaluator = QuestionEvaluator(question=question)
        evaluator.run_eval()


def entrypoint():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
