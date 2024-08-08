import requests
from pathlib import Path
import os
from llama_cpp import Llama
import chromadb

def _load_env():
    env_path = Path('../.env').resolve()
    for line in env_path.read_text().splitlines():
        k, v = line.split("=")
        os.environ[k] = v

def load_jina_reader(addr: str) -> requests.Response:
    jina_url = f"https://r.jina.ai/{addr}"
    headers = {"Authorization": f"Bearer {os.environ['JINA_API']}"}

    response = requests.get(jina_url, headers=headers)

    return response

if __name__ == "__main__":
    addr = "https://www.hoopersbeta.com/library/a2-pulley-manual-for-climbers"
    model_path = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    data_file = Path("../data/blog_1.md")
    db_file = Path("../data/chroma.db")
    llm = Llama(
        model_path,
        n_gpu_layers=15,
        embedding=True,
    )

    # cache JINA API call to file
    if not data_file.exists():
        _load_env()
        response = load_jina_reader(addr)
        full_text = response.text
        data_file.write_text(full_text)
    else:
        full_text = data_file.read_text()


    chunks = full_text.split("\n\n")
    ids = [f"id{i}" for i in range(len(chunks))]
    client = chromadb.PersistentClient(str(db_file.resolve()))
    collection = client.create_collection(name="blog_1")
    collection.add(documents=chunks, ids=ids)

    results = collection.query(
             query_texts=["What should I do with an A2 pulley injury?"],
             n_results=5
            )


