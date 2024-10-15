from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from loguru import logger

from dragtor import data, llm

lsh = llm.LlamaServerHandler.from_config()
logger.info(f"---\nModel:\n{lsh.modelpath}")


root_dir = Path("/Users/mischa/Projects/local/dragtor/")
json_grammar_file = root_dir / "scripts" / "json.gnbf"
json_arr_grammar_file = root_dir / "scripts" / "json_arr.gnbf"

stored_topics = {}
topics_file = root_dir / "data" / "features" / "topics.json"
stored_questions = {}
questions_file = root_dir / "data" / "features" / "questions.json"
stored_answers = {}
answers_file = root_dir / "data" / "features" / "answers.json"
failed_parse_file = root_dir / "data" / "features" / "failed_parse.txt"
for file in (
    topics_file,
    questions_file,
    answers_file,
    failed_parse_file,
):
    file.parent.mkdir(exist_ok=True, parents=True)
    file.touch()


@dataclass
class Messages:
    _parts: list = field(default_factory=list)

    def system(self, message: str):
        self._parts.append(("system", message))

    def user(self, message: str):
        self._parts.append(("user", message))

    def assistant(self, message: str):
        self._parts.append(("assistant", message))

    def format(self) -> list[dict]:
        messages = [{"role": role, "content": message} for role, message in self._parts]
        return messages


def clean_answer(answer: str) -> str:
    start_pos = answer.find("{")
    end_pos = answer.rfind("}")
    answer = f"[{answer[start_pos : end_pos + 1]}]"

    return answer


def parse_llm_json(llm_answer: str) -> list[dict]:
    llm_answer = clean_answer(llm_answer)

    json_prompt = """
    To ensure proper JSON schema formatting for input to a large language model, follow these rules:
    use double quotes for all keys and string values, escape any double quotes within string values with a backslash (\\), separate key-value pairs with commas, enclose objects in curly braces ({{}}), and arrays in square brackets ([]).
    Ensure all keys are unique within the same object, values can be strings, numbers, objects, arrays, true, false, or null.
    Maintain proper nesting and closure of braces and brackets.
    Avoid trailing commas after the last key-value pair or array item.
    Use UTF-8 encoding and ensure the entire JSON is a single valid structure without extraneous characters.
    The following JSON string is invalid. Fix it. {e}
    {failing_json}
    """
    try:
        parsed = json.loads(llm_answer)
    except json.JSONDecodeError as e:
        logger.error("Failing to parse JSON output, trying to correct")
        logger.error(e)
        correcting_messages = Messages()
        correcting_messages.user(json_prompt.format(e=e, failing_json=llm_answer))
        corrected_answer = lsh.chat_llm(correcting_messages.format(), n_predict=2048)
        try:
            parsed = json.loads(clean_answer(corrected_answer))
        except json.JSONDecodeError:
            logger.error("Failing to parse corrected JSON output")
            logger.error(e)
            logger.error(corrected_answer)
            with failed_parse_file.open("a") as f:
                f.write("\n---\n")
                f.write(corrected_answer)
                f.write("\n---\n")
            parsed = json.loads("[{}]")

    return parsed


def extract_topics(text: str) -> list[dict]:
    sys_prompt1 = """You recieve a document and your task is to find the 2-6 most important topics which are discussed in it.
    The document will be contained in <Document> and </Document>.
    For each of the identified topics that are covered in the document you name the topic and write a description of the topic.
    The description should contain one sentence summarizing what the topic is about and three concise sentences summarizing the key insights from the document.

    Use the following JSON format for your output:
    [
    {
      "topic": "<text>",
      "description": "<text>",
    },
    ...
    ]

    Write only the JSON output in your answer!
    """

    user_prompt1 = """
    Please describe the main topics in the following document:
    <Document>
    {text}
    </Document>

    Covered Topics:

    """

    messages = Messages()
    messages.system(sys_prompt1)
    messages.user(user_prompt1.format(text=text))

    # start server externally to make experimentation faster
    # with lsh:
    llm_answer = lsh.chat_llm(
        messages.format(),
        cache_prompt=True,
        n_predict=2048,
        grammar_file=str(json_arr_grammar_file.resolve()),
    )

    topics = parse_llm_json(llm_answer)

    logger.info(f"---\nParsed Topics:\n{topics}")

    for topic_block in topics:
        assert "topic" in topic_block
        assert "description" in topic_block

    return topics


def create_questions_from_topic(topic_block: dict) -> list[dict]:
    sys_prompt2 = """You are an expert on climbing related health and training topics.
    You recieve a topic which is discussed in an unknown document and a description of that topic.
    Your task is to formulate 5 questions that climbers could ask on that topic.
    The questions should search to expand points mentioned in the description of the topics.
    The topic will be enclosed in <Topic> and </Topic> and the description of the topic will be enclosed in <TopicDescription> and </TopicDescription>.

    Use the following JSON format for your output:
    [
    {
        "question_id": "<#>",
        "question": "<text>",
    },
    ...
    ]

    Write only the JSON output in your answer!
    """

    user_prompt2 = """
    <Topic>
    {topic}
    </Topic>

    <TopicDescription>
    {description}
    </TopicDescription>

    Questions:

    """

    messages = Messages()
    messages.system(sys_prompt2)
    messages.user(
        user_prompt2.format(
            text=text, topic=topic_block["topic"], description=topic_block["description"]
        )
    )

    # start server externally to make experimentation faster
    # with lsh:
    answer2 = lsh.chat_llm(
        messages.format(), cache_prompt=True, grammar_file=str(json_arr_grammar_file.resolve())
    )
    logger.info(f"Creating questions for topic {topic_block['topic']}:")

    questions = parse_llm_json(answer2)

    logger.info(f"---\nQuestions:\n{questions}")

    return questions


def answer_question_from_text(question: str, answer: str) -> str:
    sys_prompt = """You are an expert on climbing related health and training topics.
    You recieve a document with reference information and a question regarding the content of that document.
    Your task is to answer the question using only information available from the document.
    You answer in two paragraphs of concise language, using only facts that are found in the document.
    When there is no relevant information in the document to answer the question, then answer "No relevant information found".
    The document will be enclosed in <Document> and </Document>.
    """

    user_prompt = """
    <Document>
    {text}
    </Document>

    {question}
    """

    messages = Messages()
    messages.system(sys_prompt)
    messages.user(user_prompt.format(text=text, question=question))

    # start server externally to make experimentation faster
    logger.info(f"Answering question: {question}")
    answer = lsh.chat_llm(messages.format(), cache_prompt=True)

    logger.info(f"---\nAnswer:\n{answer}")

    return answer


if __name__ == "__main__":
    do_short_test = True
    do_generate_topics = True
    do_generate_questions = True
    do_generate_answers = True

    dl = data.JinaLoader()
    full_texts = dl.get_cache()
    logger.info(f"{len(full_texts)} cached texts available")

    # for testing:
    if do_short_test:
        full_texts = full_texts[:1]
        logger.info("using only a single text for testing")

    if do_generate_topics:
        for text in full_texts:
            topics = extract_topics(text)

            text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            stored_topics[text_id] = topics
            topics_file.write_text(json.dumps(stored_topics))
    else:
        stored_topics = json.loads(topics_file.read_text())

    if do_generate_questions:
        for text in full_texts:
            text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            topics = stored_topics[text_id]

            stored_questions[text_id] = {}
            for topic_block in topics:
                questions = create_questions_from_topic(topic_block)

                topic_id = topic_block["topic"]

                stored_questions[text_id][topic_id] = questions
                questions_file.write_text(json.dumps(stored_questions))
    else:
        stored_questions = json.loads(questions_file.read_text())

    if do_generate_answers:
        for text in full_texts:
            text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            topics = stored_topics[text_id]

            stored_answers[text_id] = {}
            for topic_block in topics:
                topic_id = topic_block["topic"]
                questions = stored_questions[text_id][topic_id]

                stored_answers[text_id][topic_id] = {}
                for question in questions:
                    q_text = question["question"]
                    answer = answer_question_from_text(q_text, text)

                    stored_answers[text_id][topic_id][q_text] = answer
                    answers_file.write_text(json.dumps(stored_answers))

# TODO: check how to enforce structured output
