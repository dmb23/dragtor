"""This is an example script to present the most basic workflow of this (or any) RAG application.

This is not menat to be used, just to get an idea what is happening!
"""

import hashlib

import chromadb
from chromadb.utils import embedding_functions
from dragtor.data import JinaLoader
from llama_cpp import Llama

full_texts = JinaLoader().get_cache()[:1]

chunks = [chunk for text in full_texts for chunk in text.split("\n\n")]
chunks = list(set(chunks))
embeddings = embedding_functions.DefaultEmbeddingFunction()(chunks)
ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]

client = chromadb.Client()
collection = client.create_collection(
    "test_collection",
    metadata={"hnsw:space": "l2"},
)
collection.add(ids=ids, embeddings=embeddings, documents=chunks)


question = "what is the meaning of life?"

search_results = collection.query(query_texts=[question])["documents"][0]

prompt = """Use the following context to answer the final question.

context:
{context}

question:
{question}

answer:
""".format(context="\n".join(search_results[:2]), question=question)

llm = Llama("./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

answer = llm(prompt, max_tokens=512)

print(answer)
