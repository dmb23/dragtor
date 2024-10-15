"""What should happen?

- write functionality to get propositions from an answer in a list
- get propositions from an answer
- check how many of those are grounded in the context

Also interesting:
- get propositions from a reference answer
- check how many TP / FP / FN I get?
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from dragtor import data, llm

lsh = llm.LlamaServerHandler.from_config()
logger.info(f"---\nModel:\n{lsh.modelpath}")

root_dir = Path("/Users/mischa/Projects/local/dragtor/")
answers_file = root_dir / "data" / "features" / "02_answers.json"


def ident(text: str) -> str:
    """Create a unique key for a text"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


class Propositions(BaseModel):
    propositions: list[str]


def get_propositions(text: str) -> Propositions:
    sys_prompt = """
    Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
    3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
    4. Present the results as a list of strings, formatted in JSON. Return only the JSON output.
    """

    user_prompt_template = """Decompose the following:
    <content>
    {content}
    </content>
    """

    ex_in = user_prompt_template.format(
        content="""The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny." """
    )
    ex_out = json.dumps(
        [
            "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.",
            "Georg Franck von Franckenau was a professor of medicine.",
            "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.",
            "Richard Sermon was a scholar.",
            "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter",
            "Hares were frequently seen in gardens in spring.",
            "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.",
            "There is a European tradition that hares laid eggs.",
            "A hare’s scratch or form and a lapwing’s nest look very similar.",
            "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.",
            "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.",
            "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.",
            "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America.",
        ]
    )

    messages = Messages()
    messages.system(sys_prompt)
    messages.user(ex_in)
    messages.assistant(ex_out)
    messages.user(user_prompt_template.format(content=text))

    logger.info(f'decomposing section " {text[:40]} [...] {text[-40:]}"')

    answer = lsh.chat_llm(
        messages.format(), cache_prompt=True, json_schema=Propositions.model_json_schema()
    )

    propositions = Propositions.model_validate_json(answer)

    return propositions


class Faithfulness(BaseModel):
    statement: str
    reasoning: str
    is_inferred: bool


class EvalFaithful(BaseModel):
    evals: list[Faithfulness]


def get_faithfullness(props: Propositions, context: str) -> EvalFaithful:
    def _format_user_message(statements: list[str], context: str) -> str:
        return json.dumps({"context": context, "statements": statements})

    sys_prompt = """
    Your task is to judge the faithfulness of a series of statements based on a given context.
    For each statement you must return is_inferred as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
    For each statement, return the statement, the reasoning and the value for is_inferred in JSON.
    Output only the final JSON array, nothing else.
    """

    ex_in = _format_user_message(
        context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
        statements=[
            "John is majoring in Biology.",
            "John is taking a course on Artificial Intelligence.",
            "John is a dedicated student.",
            "John has a part-time job.",
        ],
    )

    ex_out = EvalFaithful(
        evals=[
            Faithfulness(
                statement="John is majoring in Biology.",
                reasoning="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                is_inferred=False,
            ),
            Faithfulness(
                statement="John is taking a course on Artificial Intelligence.",
                reasoning="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                is_inferred=False,
            ),
            Faithfulness(
                statement="John is a dedicated student.",
                reasoning="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                is_inferred=True,
            ),
            Faithfulness(
                statement="John has a part-time job.",
                reasoning="There is no information given in the context about John having a part-time job.",
                is_inferred=False,
            ),
        ]
    )

    messages = Messages()
    messages.system(sys_prompt)
    messages.user(ex_in)
    messages.assistant(ex_out.model_dump_json())
    messages.user(_format_user_message(statements=props.propositions, context=context))

    answer = lsh.chat_llm(
        messages.format(),
        cache_prompt=True,
        json_schema=EvalFaithful.model_json_schema(),
        n_predict=2048,
    )

    ef = EvalFaithful.model_validate_json(answer)

    return ef


if __name__ == "__main__":
    dl = data.JinaLoader()
    full_texts = dl.get_cache()
    logger.info(f"{len(full_texts)} cached texts available")

    contexts = {ident(text): text for text in full_texts}

    answers = json.loads(answers_file.read_text())
    text_key, topic_answers = next(iter(answers.items()))
    topic, questions = next(iter(topic_answers.items()))
    question, answer = next(iter(questions.items()))

    if False:
        props = get_propositions(answer)
    else:
        props = Propositions(
            propositions=[
                "According to the document, weight training can help rock climbers improve their strength.",
                "Weight training can help climbers learn how to control tension.",
                "Weight training can help climbers build body awareness.",
                "Weight training can help climbers improve their ability to pull harder.",
                "Weight training can help climbers improve their strength without adding weight.",
                "The document provides an example of the deadlift exercise.",
                "The deadlift exercise is a good example of body awareness and tension control.",
                "The deadlift forces the climber to engage their scapular retractors.",
                "The deadlift forces the climber to engage their core.",
                "The deadlift forces the climber to engage their glutes.",
                "The deadlift forces the climber to engage their hamstrings.",
                "Strength training can help improve the force capacity of muscles.",
                "Strength training can help improve the force capacity of the shoulders.",
                "Strength training can help the muscles sustain higher levels of force before failing.",
                "Weight training can help climbers improve their strength in specific muscle groups.",
                "The document suggests that weight training can be a valuable tool for rock climbers looking to improve their strength.",
                "Weight training can help climbers improve their ability to pull harder.",
                "Weight training can help climbers build body awareness.",
                "Weight training can help climbers control tension.",
                "Incorporating weight training into their training program can help climbers improve their climbing performance.",
            ]
        )

    logger.info(props)

    context = contexts[text_key]

    eval_faithful = get_faithfullness(props, context)

    logger.info(eval_faithful)
