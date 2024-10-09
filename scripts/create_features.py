from dataclasses import dataclass, field
import json

from loguru import logger

from dragtor import data, llm

dl = data.JinaLoader()
# index = get_index()
lsh = llm.LlamaServerHandler.from_config()

full_texts = dl.get_cache()
text = full_texts[0]

logger.info(f"---\nModel:\n{lsh.modelpath}")


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


do_first_step = False
if do_first_step:
    sys_prompt1 = """You recieve a document and your task is to find the 2-6 most important topics which are discussed in it.
    The document will be contained in <Document> and </Document>.
    For each of the identified topics that are covered in the document you name the topic and write one sentence in which you give a concise description of the topic.

    Use the following JSON format for your output:
    {
      "topic": "<text>",
      "description": "<text>",
    },

    Write only the JSON output in your answer!
    """

    user_prompt1 = """
    Please describe the main topics in the following document:
    <Document>
    {text}
    </Document>

    Covered Topics:

    """
    logger.info(f"---\nSystem Prompt:\n{sys_prompt1}")
    logger.info(f"---\nUser Prompt:\n{user_prompt1}")

    messages = Messages()
    messages.system(sys_prompt1)
    messages.user(user_prompt1.format(text=text))

    # start server externally to make experimentation faster
    # with lsh:
    answer1 = lsh.chat_llm(messages.format(), cache_prompt=True)
    logger.info(f"---\nAnswer:\n{answer1}")

else:
    answer1 = """Answer:
{
  "name": "Understanding the Anatomy of the A2 Pulley",
  "description": "This topic delves into the anatomy of the A2 pulley, explaining how it works and its importance in allowing climbers to hold onto edges without the other parts of their finger coming apart."
},
{
  "name": "Causes of A2 Pulley Injury",
  "description": "This topic discusses the causes of A2 pulley injury, including the crimping position, shock loading, repetitive strain, and overtraining."
},
{
  "name": "Testing and Evaluation of A2 Pulley Injury",
  "description": "This topic covers the testing and evaluation of A2 pulley injury, including observation, palpation, range of motion, and tissue loading to determine the severity of the injury."
},
{
  "name": "Treatment and Rehabilitation of A2 Pulley Injury",
  "description": "This topic discusses the treatment and rehabilitation of A2 pulley injury, including active range of motion, soft tissue mobilization, diet and sleep, exercise, H-taping, and surgery."
},
{
  "name": "Prevention of A2 Pulley Injury",
  "description": "This topic provides tips and strategies for preventing A2 pulley injury, including training for tissue adaptation, getting proper rest and avoiding overtraining, proper sleep, diet, and hydration, and following the Rule of 7."
},"""


def clean_answer(answer: str) -> str:
    start_pos = answer.find("{")
    end_pos = answer.rfind("}")
    answer = f"[{answer[start_pos : end_pos + 1]}]"

    return answer


do_second_step = False
if do_second_step:
    answer1 = clean_answer(answer1)
    logger.info(f"---\nStripped Answer:\n{answer1}")

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
        topics = json.loads(answer1)
    except json.JSONDecodeError as e:
        logger.error("Failing to parse JSON output, trying to correct")
        logger.error(e)
        correcting_messages = Messages()
        correcting_messages.user(json_prompt.format(e=e, failing_json=answer1))
        corrected_answer = lsh.chat_llm(correcting_messages.format())
        try:
            topics = json.loads(answer1)
        except json.JSONDecodeError:
            logger.error("Failing to parse corrected JSON output")
            logger.error(e)
            logger.error(corrected_answer)
            topics = json.loads("[]")

    logger.info(f"---\nParsed Answer:\n{topics}")
else:
    topics = json.loads(
        """[{"topic": "Anatomy of the A2 Pulley", "description": "The A2 pulley is one of five pulleys in each finger that holds the flexor tendon tight up against the bones, allowing for flexion of the finger without the tendon bowstringing out."}, {"topic": "Causes of A2 Pulley Injuries", "description": "A2 pulley injuries are caused by exceeding the limit of the pulley, which can happen when subjecting the fingers to extraordinary forces, such as crimping, or through repetitive stress and overtraining."}, {"topic": "Testing and Diagnosis of A2 Pulley Injuries", "description": "A2 pulley injuries can be diagnosed through observation, palpation, range of motion, and tissue loading tests, which can help determine the severity of the injury."}, {"topic": "Treatment and Rehabilitation of A2 Pulley Injuries", "description": "Treatment of A2 pulley injuries involves active range of motion, soft tissue mobilization, diet and sleep, exercise, and retraining, which can help promote healing and prevent further injury."}, {"topic": "Prevention of A2 Pulley Injuries", "description": "Prevention of A2 pulley injuries involves training for tissue adaptation, getting proper rest and avoiding overtraining, proper sleep, diet, and hydration, following the rule of 7, and climbing different types of holds and styles."}]"""
    )

for topic_block in topics:
    assert "topic" in topic_block
    assert "description" in topic_block

topic_block = topics[3]

sys_prompt2 = """You recieve a document and a topic which is discussed in the document and a description of that topic.
Your task is to formulate 5 questions that can be asked on that topic that can be answered by the information provided in the document.
The answer to each question should be fully contained in the document, and not rely on external knowledge.
The document will be enclosed in <Document> and </Document>, the topic will be enclosed in <Topic> and </Topic>, the description of the topic will be enclosed in <TopicDescription> and </TopicDescription>.

Use the following JSON format for your output:
{
    "question_id": "<#>",
    "question": "<text>",
},

Write only the JSON output in your answer!
"""


user_prompt2 = """
<Document>
{text}
</Document>

<Topic>
{topic}
</Topic>

<TopicDescription>
{description}
</TopicDescription>

Questions:

"""
logger.info(f"---\nSystem Prompt:\n{sys_prompt2}")
logger.info(f"---\nUser Prompt:\n{user_prompt2}")

messages = Messages()
messages.system(sys_prompt2)
messages.user(
    user_prompt2.format(
        text=text, topic=topic_block["topic"], description=topic_block["description"]
    )
)

# start server externally to make experimentation faster
# with lsh:
answer2 = lsh.chat_llm(messages.format(), cache_prompt=True)
logger.info(f"Creating questions for topic {topic_block['topic']}:")
logger.info(f"---\nQuestions:\n{answer2}")
