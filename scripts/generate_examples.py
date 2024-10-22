"""Use the groq API to generate gold-truth answers to specific questions on source material

requires a Groq API key in config.creds.groq
"""

import json
from pathlib import Path
from time import sleep

from loguru import logger
import requests

from dragtor import config, utils

groq_model = "llama-3.1-70b-versatile"


def groq_the_question(question: str, text: str, n_repeat: int = 0) -> str:
    system_prompt = """
    You are an helpful assistant that extracts information on climbing training and climbing-related injury diagnosis and treatment from long-form text sources.
    You recieve a question together with in-depth background information.
    You answer the questions solely based on the provided context.
    If there is no information provided in the context to answer the question, you answer "There is no relevant information provided".
    You provide answers in three paragraphs of text explaining the key concepts from the referenced text.
    You formulate all statements as direct propositions without referencing the source, i.e. "It is good to train" instead of "The text says it is good to train".
    """

    prompt = """Use the following context to answer the question at the end:

    <context>:
    {context}
    </context>

    {question}

    """.format(context=text, question=question)

    messages = utils.Messages()
    messages.system(system_prompt)
    messages.user(prompt.format(context=text, question=question))

    data = {"messages": messages.format(), "model": groq_model}
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.conf.creds.groq}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)
    logger.debug(response.status_code)
    logger.debug(response.json())

    match response.status_code:
        case 200:
            answer = response.json()["choices"][0]["message"]["content"]
            logger.info(answer)

            return answer
        case 429:
            if n_repeat >= 2:
                logger.info("Rate limit exceeded. But I have waited already soooooo long!")
                return ""
            logger.info(f"Rate limit exceeded! Waiting 30sec...({n_repeat})")
            sleep(30)
            return groq_the_question(question=question, text=text, n_repeat=n_repeat + 1)
        case _:
            logger.warning(f"What is happening? Return Code {response.status_code}")
            return ""


if __name__ == "__main__":
    blog_questions = {
        "jina_https:__www.hoopersbeta.com_library_weight-training-and-rock-climbing.md": "What are the main reasons to weight train as a climber?",
        "jina_https:__www.hoopersbeta.com_library_how-to-heal-from-a-lumbrical-injury-5-simple-stages-to-recover.md": "What are progressive strength exercises when rehabbing a lumbrical injury?",
        "jina_https:__www.hoopersbeta.com_library_flexor-tenosynovitis.md": "How can I find out if pain at a pulley comes from a pulley injury or from a different injury?",
        "jina_https:__www.hoopersbeta.com_library_a2-pulley-manual-for-climbers.md": "What are possible steps to rehab an A2 pulley injury?",
        "jina_https:__www.hoopersbeta.com_library_will-hangboarding-2x_day-improve-your-climbing-ultimate-revised-breakdown.md": "Should I hangboard 2 times per day to increase my finger strength?",
    }

    base_path = Path(config.conf.base_path)
    jina_cache = base_path / config.conf.data.jina_cache

    reference_file = base_path / config.conf.eval.eval_dir / config.conf.eval.eval_answers
    if reference_file.is_file():
        gold_references = json.loads(reference_file.read_text())
    else:
        gold_references = {}

    for filename, question in blog_questions.items():
        logger.info(f"Question: {question}")
        if question not in gold_references:
            text = (jina_cache / filename).read_text()

            answer = groq_the_question(question, text)
            if answer:
                gold_references[question] = answer
                reference_file.write_text(json.dumps(gold_references))
            logger.info("question sent to Groq API")
        else:
            logger.info("question already answered in reference")
