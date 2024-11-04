from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from loguru import logger
import torch
from transformers import AutoModel, AutoTokenizer

from dragtor import config
from dragtor.index import RetrievalError
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import ChromaDBStore, get_store
from dragtor.utils import ident


@dataclass
class Index(ABC):
    chunker: Chunker
    store: ChromaDBStore
    reranker: Reranker

    @abstractmethod
    def index_texts(self, texts: list[str]) -> None:
        pass

    def query(self, question: str) -> list[str]:
        n_results = config.conf.select("store.n_query", default=5)
        n_ranked = config.conf.select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        logger.debug(
            f"extracting {n_results} documents from the store, then rerank down to {n_ranked}"
        )
        found_docs = self.store.query(question, n_results)
        logger.debug(f"obtained {len(found_docs)} documents")
        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"obtained {len(ranked_docs)} after reranking")
            return ranked_docs
        return found_docs

    @classmethod
    def from_defaults(cls) -> Self:
        chunker = get_chunker()
        store = get_store()
        reranker = get_reranker()
        logger.debug("initialized all parts of Index")
        return cls(chunker, store, reranker)


@dataclass
class BasicIndex(Index):
    def index_texts(self, texts: list[str]) -> None:
        chunks = self.chunker.chunk_texts(texts)
        self.store.add_chunks(chunks)


class LateChunkingIndex(Index):
    # TODO: needs more debugging, priorities shifted
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("WIP, needs to be finished!")
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        if config.conf.select("embeddings.strategy") != "jina":
            logger.warning(
                f"Configured embeddings are {config.conf.select('embeddings.strategy')} but LateChunkingIndex always uses Jina embeddings!"
            )

    def index_texts(self, texts: list[str]) -> None:
        all_chunks, all_ids, all_embeddings = [], [], []
        for text in texts:
            chunks, _ = self.chunker.chunk_and_annotate(text)
            embeddings = self._calculate_late_embeddings_v2(text)
            ids = [ident(chunk) for chunk in chunks]

            all_chunks.extend(chunks)
            all_ids.extend(ids)
            all_embeddings.extend(embeddings)

        self.store.collection.add(documents=all_chunks, ids=all_ids, embeddings=all_embeddings)

    def _calculate_late_embeddings_v2(self, input_text: str):
        # TODO: check the logic around a max length
        # what happens when I want to embed a text that is longer than my embedding model can handle?
        chunks, _ = self.chunker.chunk_and_annotate(input_text)

        inputs = self.tokenizer(input_text, return_tensors="pt")
        chunk_tokens = [
            self.tokenizer(chunk, add_special_tokens=False, return_tensors="pt")["input_ids"]
            for chunk in chunks
        ]
        if not (inputs["input_ids"][:, 1:-1] == torch.cat(chunk_tokens, 1)).all():
            raise ValueError("Something is fishy with text / chunk embeddings!")
        i_start = 1
        chunk_token_annotations = []
        for chunk in chunk_tokens:
            i_end = i_start + chunk.shape[1]
            chunk_token_annotations.append((i_start, i_end))
            i_start = i_end

        model_output = self.model(**inputs)
        token_embeddings = model_output["last_hidden_state"].squeeze(0)
        pooled_embeddings = [
            token_embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in chunk_token_annotations
        ]
        pooled_embeddings = [embedding.detach().cpu().tolist() for embedding in pooled_embeddings]

        return pooled_embeddings

    def query(self, question: str) -> list[str]:
        n_results = config.conf.select("store.n_query", default=5)
        n_ranked = config.conf.select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        logger.debug(
            f"extracting {n_results} documents from the store, then rerank down to {n_ranked}"
        )
        q_embedding = self.model.encode(question)
        search_result = self.store.collection.query(
            query_embeddings=[q_embedding], n_results=n_results
        )
        try:
            found_docs = search_result["documents"][0]
            logger.debug(f"obtained {len(found_docs)} documents")
        except TypeError:
            raise RetrievalError("Could not find any document in the index!")

        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"obtained {len(ranked_docs)} after reranking")
            return ranked_docs
        return found_docs

    # def _calculate_late_embeddings(
    #     self, input_text: str, token_chunk_annotations: list[tuple[int, int]], max_length: int = -1
    # ) -> list:
    #     inputs = self.tokenizer(input_text, return_tensors="pt")
    #     model_output = self.model(**inputs)
    #
    #     token_embeddings = model_output["last_hidden_state"]
    #     outputs = []
    #     if max_length > 0:  # remove annotations which go bejond the max-length of the model
    #         token_chunk_annotations = [
    #             (start, min(end, max_length - 1))
    #             for (start, end) in token_chunk_annotations
    #             if start < (max_length - 1)
    #         ]
    #     pooled_embeddings = [
    #         token_embeddings[start:end].sum(dim=0) / (end - start)
    #         for start, end in token_chunk_annotations
    #         if (end - start) >= 1
    #     ]
    #     pooled_embeddings = [embedding.detach().cpu().numpy() for embedding in pooled_embeddings]
    #     outputs.append(pooled_embeddings)
    #
    #     return outputs


def get_index() -> Index:
    strat = config.conf.select("index.strategy", "default")
    match strat:
        case "default" | "basic":
            return BasicIndex.from_defaults()
        case "late_chunking":
            return LateChunkingIndex.from_defaults()
        case _:
            raise ValueError(f"Uknown value '{strat}' for index.strategy!")