"""An 'Index' is a class that manages all interaction with possible context information.

Main entrypoints:
- index_texts: add information from new texts to the index
- query: obtain relevant information for a prompt / question

I want to keep indexing very flexible. Funcitonality can vary:
- add parts of the original texts verbatim and obtain the most relevant ones
- employ more complicated embedding schemes, which require more information than just the text
- create additional information (summaries, possible questions) to store on top of the texts
- obtain the full document that contains a matching snippet
- ...

NOTE: an Index could contain in the future agentic steps after retrieval.
At that point it provides an interesting additional layer of abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import chromadb.api
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from dragtor import config
from dragtor.config import conf
from dragtor.index import RetrievalError
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import VectorStore, get_store
from dragtor.utils import ident


class Index(ABC):
    @abstractmethod
    def index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        pass

    @abstractmethod
    def query(self, question: str) -> list[str]:
        pass


@dataclass
class BasicIndex(Index):
    store: VectorStore
    reranker: Reranker

    def index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self.store.add_documents(texts, metadata)

    def query(self, question: str) -> list[str]:
        n_results = config.conf.select("store.n_query", default=5)
        n_ranked = config.conf.select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        found_docs = self.store.query(question, n_results)
        logger.debug(f"extracted {len(found_docs)} documents")
        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"reranked down to {len(ranked_docs)} documents")
            return ranked_docs
        return found_docs

    @classmethod
    def from_defaults(cls) -> Self:
        store = get_store()
        reranker = get_reranker()
        logger.debug("initialized all parts of Index")
        return cls(store, reranker)


def get_index() -> Index:
    strat = config.conf.select("index.strategy", "default")
    match strat:
        case "default" | "basic":
            return BasicIndex.from_defaults()
        case "late_chunking":
            return LateChunkingIndex()
        case _:
            raise ValueError(f"Uknown value '{strat}' for index.strategy!")


@dataclass
class LateChunkingIndex(Index):
    """Idea: something
    Combines different functionality:
    - creating embeddings via late chunking for a VectorStore
    - select chunking strategy
    - create additional feature to store
    - select retrieval strategy
        - parent/child, full document, ...

    API:
    - index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None
    - query(self, question: str) -> list[str]:

    Necessary steps:
    - embed_chunks: document + chunk annotations -> chunk embeddings
        - jina v3
        - torch / transformers
    - embed_query: query -> query embedding
    - chunk document: document -> chunk annotations
    - retrieve_documents: strategy, query -> documents
    - rerank: query, documents -> documents
    """

    client: chromadb.api.ClientAPI = field(init=False)
    _db_path: str = field(init=False)
    _distance: str = field(init=False)
    tokenizer: PreTrainedTokenizer = field(init=False)
    embedding_model: PreTrainedModel = field(init=False)
    chunker: Chunker = field(default_factory=get_chunker)
    reranker: Reranker = field(default_factory=get_reranker)

    chunk_collection_name: str = "chunks"

    def __post_init__(self):
        self._db_path = str(Path(conf.base_path) / conf.store.chromadb.path)
        self._distance = conf.select("store.chromadb.distance", default="cosine")
        self.client = chromadb.PersistentClient(path=self._db_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
        self.embedding_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )

    def collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": self._distance,
            },
        )

    def _map_chunks_to_tokens(
        self, chunk_annotations: list[tuple[int, int]], token_offsets: np.ndarray
    ) -> list[tuple[int, int]]:
        """Translate chunk boundaries from text to token space"""
        logger.debug(f"Mapping {len(chunk_annotations)} chunks onto {len(token_offsets)} tokens")
        # Convert chunk annotations to NumPy arrays
        ann = np.array(chunk_annotations)

        # Find start token indices
        start_conditions = (token_offsets[:, 0][:, np.newaxis] <= ann[:, 0]) & (
            token_offsets[:, 1][:, np.newaxis] > ann[:, 0]
        )
        start_tokens = np.argmax(start_conditions, axis=0)

        # Find end token indices
        end_conditions = (token_offsets[:, 0][:, np.newaxis] < ann[:, 1]) & (
            token_offsets[:, 1][:, np.newaxis] >= ann[:, 1]
        )
        end_tokens = np.argmax(end_conditions, axis=0) + 1  # end_token is exclusive

        # Check for any unmapped boundaries
        if not np.all(start_conditions[start_tokens, np.arange(len(chunk_annotations))]):
            raise ValueError("Could not map some chunk boundaries to tokens.")
        if not np.all(end_conditions[end_tokens - 1, np.arange(len(chunk_annotations))]):
            raise ValueError("Could not map some chunk boundaries to tokens.")

        token_chunk_annotations = list(zip(start_tokens, end_tokens))
        return token_chunk_annotations

    def _get_long_token_embeddings(self, model_inputs):
        task = "retrieval.passage"
        task_id = self.embedding_model._adaptation_map[task]

        max_seq_length = config.conf.get("index.late_chunking.long_seq_length", 2048)

        # only single string `input_text`
        adapter_mask = torch.full((1,), task_id, dtype=torch.int32)

        if len(model_inputs["token_ids"]) <= max_seq_length:
            with torch.no_grad():
                model_output = self.embedding_model(**model_inputs, adapter_mask=adapter_mask)

            token_embeddings = model_output["last_hidden_state"].squeeze(0).float()

        else:
            # Split tokens into overlapping chunks
            overlap = config.conf.get("index.late_chunking.long_seq_overlap", 256)
            chunks = []
            chunk_embeddings = []

            # Create chunks with overlap
            for i in range(0, len(model_inputs["token_ids"]), max_seq_length - overlap):
                chunk = {
                    k: v[:, i : i + max_seq_length] if isinstance(v, torch.Tensor) else v
                    for k, v in model_inputs.items()
                }
                chunks.append(chunk)

            # Get embeddings for each chunk
            for chunk in chunks:
                with torch.no_grad():
                    chunk_output = self.embedding_model(**chunk, adapter_mask=adapter_mask)
                chunk_embeddings.append(chunk_output["last_hidden_state"].squeeze(0).float())

            # Combine embeddings from overlapping regions by averaging
            # NOTE: reference implementation of Jina only uses the embeddings of the later chunk
            # https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/mteb_chunked_eval.py#L128
            token_embeddings = torch.zeros(
                (len(model_inputs["token_ids"][0]), chunk_embeddings[0].shape[1]),
                dtype=chunk_embeddings[0].dtype,
            )

            counts = torch.zeros(len(model_inputs["token_ids"][0]), dtype=torch.int)

            pos = 0
            for chunk_emb in chunk_embeddings:
                chunk_size = chunk_emb.shape[0]
                token_embeddings[pos : pos + chunk_size] += chunk_emb
                counts[pos : pos + chunk_size] += 1
                pos += max_seq_length - overlap

            # Average overlapping regions
            token_embeddings = token_embeddings / counts.unsqueeze(1)

        return token_embeddings

    def _calculate_late_embeddings(
        self, input_text: str, chunk_annotations: list[tuple[int, int]]
    ) -> list[list[float]]:
        task = "retrieval.passage"
        task_prefix = self.embedding_model._task_instructions[task]

        inputs = self.tokenizer(
            task_prefix + input_text, return_tensors="pt", return_offsets_mapping=True
        )

        token_embeddings = self._get_long_token_embeddings(inputs["input_ids"])

        # task prefix was added for Jina v3, correct for that
        token_offsets = inputs["offset_mapping"].squeeze(0).numpy() - len(task_prefix)
        token_chunk_annotations = self._map_chunks_to_tokens(chunk_annotations, token_offsets)

        pooled_embeddings = [
            token_embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in token_chunk_annotations
        ]
        pooled_embeddings = [
            F.normalize(embedding, p=2, dim=0).detach().cpu().tolist()
            for embedding in pooled_embeddings
        ]

        return pooled_embeddings

    def index_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        all_chunks, all_ids, all_embeddings = [], [], []
        for text in texts:
            chunks, chunk_annotations = self.chunker.chunk_and_annotate(text)

            embeddings = self._calculate_late_embeddings(text, chunk_annotations)

            ids = [ident(chunk) for chunk in chunks]

            all_chunks.extend(chunks)
            all_ids.extend(ids)
            all_embeddings.extend(embeddings)

        # Store the chunks, ids, embeddings, and token_chunk_annotations in the vector store
        self.collection(self.chunk_collection_name).add(
            documents=all_chunks,
            ids=all_ids,
            embeddings=all_embeddings,
        )

    def query(self, question: str) -> list[str]:
        task = "retrieval.query"
        n_results = conf.select("store.n_query", default=5)
        n_ranked = conf.select("reranker.n_ranked", default=3)
        _do_rerank = True
        if n_results <= n_ranked:
            n_ranked = n_results
            _do_rerank = False

        q_embedding = self.embedding_model.encode(question, task=task)
        search_result = self.collection(self.chunk_collection_name).query(
            query_embeddings=[q_embedding], n_results=n_results
        )
        try:
            found_docs = search_result["documents"][0]
            logger.debug(f"extracted {len(found_docs)} documents")
        except TypeError:
            raise RetrievalError("Could not find any document in the index!")

        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_ranked)
            logger.debug(f"reranked down to {len(ranked_docs)} documents")
            return ranked_docs
        return found_docs
