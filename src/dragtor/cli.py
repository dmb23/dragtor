"""Process Logic for dRAGtor application. Neatly packed as a CLI"""

import fire
from loguru import logger
from omegaconf import OmegaConf

from dragtor import config
from dragtor.data import get_all_loaders
from dragtor.index.index import get_index
from dragtor.llm import LocalDragtor
from dragtor.llm.evaluation import EvaluationSuite, QuestionEvaluator


class Cli:
    """Ask the dRAGtor your questions about climbing injuries!

    Fully local RAG for knowledge from various sources about climbing injuries and treatment
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            logger.debug(f"updating {k} with value {v}")
            OmegaConf.update(config.conf, k, v)

    @logger.catch
    def load(self):
        """Load remote data to local files for further processing"""
        for loader in get_all_loaders():
            loader.load_to_cache()
        logger.info("Loaded configured data successfully")

    def clear_index(self):
        """Reset all data already in the index"""
        index = get_index()

        try:
            for c in index.store.client.list_collections():  # ruff: noqa:
                index.store.client.delete_collection(c.name)
        except AttributeError:
            raise ValueError("clear_index only works for a Chroma DB Store!")

    @logger.catch
    def index(self):
        """Create a Vector Store of embeddings of all loaded sources for retrieval"""
        # full_texts = []
        # for loader in get_all_loaders():
        #     full_texts += loader.get_cache()
        full_texts = sum([l.get_cache() for l in get_all_loaders()], start=[])

        index = get_index()
        index.index_texts(full_texts)
        logger.info("Indexed all cached data successfully")

    # @logger.catch
    # def preload(self):
    #     """Create pre-loaded state files for all loaded sources for retrieval"""
    #     loader = data.JinaLoader()
    #     full_texts = loader.get_cache()
    #
    #     ld = LocalDragtor()
    #     for i, text in enumerate(full_texts):
    #         text_id = ident(text)
    #         messages = ld._to_messages(question="", context=text)
    #         filename = f"{text_id}.bin"
    #         ld.llm.store_state(messages, filename)
    #         logger.info(f"Preloaded {i+1}/{len(full_texts)} texts")

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

    def eval(self, question: str = ""):
        """Evaluate the performance of the configured RAG setup.

        - evaluate how many of the propositions in a given answer are based on the sources.
        - possibly: evaluate answers to reference questions against gold truths
        """
        if question:
            evaluator = QuestionEvaluator(question=question)
            evaluator.show_eval()
        else:
            evaluator = EvaluationSuite()
            evaluator.run_all_evals()


def entrypoint():
    """Ask the dRAGtor your questions about climbing injuries!"""
    fire.Fire(Cli)
