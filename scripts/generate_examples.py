import json
from pathlib import Path

from llama_cpp import Llama

modelfile = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
llm = Llama(
    modelfile,
    n_ctx=16000,
    n_gpu_layers=15,
)

questions = {
    "jina_https:__www.hoopersbeta.com_library_weight-training-and-rock-climbing.md": "What are the main reasons to weight train as a climber?",
    "jina_https:__www.hoopersbeta.com_library_how-to-heal-from-a-lumbrical-injury-5-simple-stages-to-recover.md": "What are progressive strength exercises when rehabbing a lumbrical injury?",
    "jina_https:__www.hoopersbeta.com_library_flexor-tenosynovitis.md": "How can I find out if pain at a pulley comes from a pulley injury or from a different injury?",
    "jina_https:__www.hoopersbeta.com_library_a2-pulley-manual-for-climbers.md": "What are possible steps to rehab an A2 pulley injury?",
    "jina_https:__www.hoopersbeta.com_library_will-hangboarding-2x_day-improve-your-climbing-ultimate-revised-breakdown.md": "Should I hangboard 2 times per day to increase my finger strength?",
}

text_files = [p for p in Path("../data/jina_reader/").glob("*.md")]

system_prompt = """You are an expert in diagnosis and treatment of climbing-related injuries. You answer questions in 3 paragraphs, where one of those paragraphs can be replaced by a numbered list.
"""

examples = []
for fpath in text_files:
    text = fpath.read_text()
    name = fpath.name
    question = questions[name]

    prompt = """Use the following context to answer the question:

context:
{context}

question:
{question}

answer:
""".format(context=text, question=question)

    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=512,
    )
    answer = result["choices"][0]["message"]["content"]

    print("-------------------------\n")
    print(f"Question:\n{question}")
    print(f"Answer:\n{answer}")

    examples.append({"name": name, "question": question, "answer": answer})

outfile = Path("./example_questions.json")
outfile.write_text(json.dumps(examples))
