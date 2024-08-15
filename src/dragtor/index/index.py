from dataclasses import dataclass, field

from dragtor.index.chunk import Chunker
from dragtor.index.rerank import Reranker, get_reranker
from dragtor.index.store import VectorStore, get_store


@dataclass
class Index:
    chunker: Chunker = field(default_factory=Chunker)
    store: VectorStore = field(default_factory=get_store)
    reranker: Reranker = field(default_factory=get_reranker)

    def index_texts(self, texts: list[str]) -> None:
        chunks = self.chunker.chunk_texts(texts)
        self.store.add_chunks(chunks)

    def query(self, question: str, n_results: int, n_extract=0) -> list[str]:
        _do_rerank = True
        if n_extract < n_results:
            n_extract = n_results
            _do_rerank = False

        found_docs = self.store.query(question, n_extract)
        if _do_rerank:
            ranked_docs = self.reranker.rerank(question, found_docs, n_results)
            return ranked_docs
        return found_docs
