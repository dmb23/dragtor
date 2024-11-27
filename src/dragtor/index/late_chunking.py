from dataclasses import dataclass, field
from pathlib import Path

import chromadb.api
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dragtor.config import conf
from dragtor.index import RetrievalError
from dragtor.index.chunk import Chunker, get_chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.utils import ident

CHUNK_COLLECTION_NAME = "chunks"


@dataclass
class LateChunkingIndex:
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

    def _calculate_late_embeddings(
        self, input_text: str, chunk_annotations: list[tuple[int, int]]
    ) -> list[list[float]]:
        # TODO: this will break if the input_text is longer than the capacity of the embedding model
        task = "retrieval.passage"
        task_id = self.embedding_model._adaptation_map[task]
        task_prefix = self.embedding_model._task_instructions[task]

        inputs = self.tokenizer(
            task_prefix + input_text, return_tensors="pt", return_offsets_mapping=True
        )

        # only single string `input_text`
        adapter_mask = torch.full((1,), task_id, dtype=torch.int32)
        with torch.no_grad():
            model_output = self.embedding_model(**inputs, adapter_mask=adapter_mask)

        token_offsets = inputs["offset_mapping"].squeeze(0).numpy()
        token_chunk_annotations = self._map_chunks_to_tokens(chunk_annotations, token_offsets)

        token_embeddings = model_output["last_hidden_state"].squeeze(0).float()

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
        self.collection(CHUNK_COLLECTION_NAME).add(
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
        search_result = self.collection(CHUNK_COLLECTION_NAME).query(
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
