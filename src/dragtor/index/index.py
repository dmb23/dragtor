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
from dragtor.data.data import Document
from dragtor.index import RetrievalError
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import VectorStore, _flatten_metadata, get_store
from dragtor.utils import ident


class Index(ABC):
    @abstractmethod
    def index_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def query(self, question: str) -> list[str]:
        pass


@dataclass
class BasicIndex(Index):
    store: VectorStore
    reranker: Reranker

    def index_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)

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
        """Translate chunk boundaries from text to token space, handling misalignments

        Semantic Segmentation needs to adjust chunk boundaries slightly,
        this can lead to misalignment with tokens.

        Especially when using late chunking, the exact boundaries of chunks in token space
        should not matter that much.
        """
        logger.debug(f"Mapping {len(chunk_annotations)} chunks onto {len(token_offsets)} tokens")
        token_chunk_annotations = []

        for chunk_start, chunk_end in chunk_annotations:
            # For start: find token whose span contains or is closest after chunk_start
            start_distances = np.where(
                token_offsets[:, 1] > chunk_start, token_offsets[:, 0] - chunk_start, np.inf
            )
            start_token = np.argmin(np.abs(start_distances))

            # For end: find token whose span contains or is closest before chunk_end
            end_distances = np.where(
                token_offsets[:, 0] < chunk_end, token_offsets[:, 1] - chunk_end, -np.inf
            )
            end_token = np.argmin(np.abs(end_distances)) + 1  # +1 for exclusive end

            # Ensure we have valid token spans
            if start_token >= end_token:
                logger.warning(
                    f"Invalid token span [{start_token}, {end_token}] for chunk [{chunk_start}, {chunk_end}]"
                )
                end_token = start_token + 1

            token_chunk_annotations.append((start_token, end_token))

        return token_chunk_annotations

    def _get_long_token_embeddings(self, model_inputs):
        """Handle inputs that might be longer than the embedding model can handle.

        Jina uses lower long-chunk sizes than the 8K tokens the model is capable of.
        Also: this method is not tested for multi-sequence inputs!
        """
        task = "retrieval.passage"
        task_id = self.embedding_model._adaptation_map[task]

        max_seq_length = config.conf.select("index.late_chunking.long_seq_length", 2048)

        # only single string `input_text`
        adapter_mask = torch.full((1,), task_id, dtype=torch.int32)

        if model_inputs["input_ids"].numel() <= max_seq_length:
            with torch.no_grad():
                model_output = self.embedding_model(**model_inputs, adapter_mask=adapter_mask)

            token_embeddings = model_output["last_hidden_state"].squeeze(0).float()

        else:
            # Split tokens into overlapping chunks
            overlap = config.conf.select("index.late_chunking.long_seq_overlap", 256)
            chunks = []
            chunk_embeddings = []

            # Create chunks with overlap
            for i in range(0, len(model_inputs["input_ids"].squeeze()), max_seq_length - overlap):
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
                (len(model_inputs["input_ids"][0]), chunk_embeddings[0].shape[1]),
                dtype=chunk_embeddings[0].dtype,
            )

            counts = torch.zeros(len(model_inputs["input_ids"][0]), dtype=torch.int)

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

        token_embeddings = self._get_long_token_embeddings(inputs)

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

    def index_documents(self, documents: list[Document]) -> None:
        all_chunks, all_ids, all_embeddings, all_metadata = [], [], [], []
        seen_ids = set()

        for doc in documents:
            # Get chunks and their positions in the text
            chunks, chunk_annotations = self.chunker.chunk_and_annotate(doc.content)

            # Calculate embeddings for chunks
            embeddings = self._calculate_late_embeddings(doc.content, chunk_annotations)

            # Generate IDs for chunks and filter duplicates
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = ident(chunk)
                if chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_embeddings.append(embedding)

                # Create metadata for the chunk
                metadata = {
                    "title": doc.title,
                    "id": doc.id,
                    "author": doc.author if doc.author else "",
                }
                if doc.metadata:
                    metadata.update(_flatten_metadata(doc.metadata))
                all_metadata.append(metadata)

        # Store everything in the vector store
        if all_chunks:  # Only add if we have chunks to store
            self.collection(self.chunk_collection_name).add(
                documents=all_chunks, ids=all_ids, embeddings=all_embeddings, metadatas=all_metadata
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
