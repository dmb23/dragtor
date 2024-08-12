import os
from pathlib import Path

import chromadb
import requests
from llama_cpp import Llama


def _load_env():
    env_path = Path("../.env").resolve()
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
    model_path = (
        "/Users/mischa/Projects/local/dragtor/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    )
    data_file = Path("/Users/mischa/Projects/local/dragtor/data/blog_1.md")
    db_file = Path("/Users/mischa/Projects/local/dragtor/data/chroma.db")
    max_tokens = 400

    llm = Llama(
        model_path,
        n_gpu_layers=15,
    )

    # cache JINA API call to file
    if not data_file.exists():
        _load_env()
        response = load_jina_reader(addr)
        full_text = response.text
        data_file.write_text(full_text)
    else:
        full_text = data_file.read_text()
    print("SCRIPT -- loaded data")

    # cache Embeddings & Vector Store
    client = chromadb.PersistentClient(path=str(db_file.resolve()))
    try:
        collection = client.get_collection(name="blog_1")
    except ValueError:
        collection = client.create_collection(name="blog_1")
        chunks = full_text.split("\n\n")
        ids = [f"id{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
    print("SCRIPT -- created VS")

    question = "What should I do with an A2 pulley injury?"
    results = collection.query(query_texts=[question], n_results=5)

    prompt = """Please use the following pieces of context to answer the question.
    context:
        {}

    question:
        {}

    answer:
    """.format(
        # '\n'.join(results['documents'][0]),
        results["documents"][0][0],
        question,
    )
    print("SCRIPT -- prompt generated")

    result = llm(
        prompt,
        max_tokens=max_tokens,
    )
    print(result["choices"][0]["text"])
#    for generated in llm(
#            prompt,
#            max_tokens=max_tokens,
#            echo=True,
#            stream=True,
#            ):
#        print(generated['choices'][0]['text'], end="")
#        sys.stdout.flush()
