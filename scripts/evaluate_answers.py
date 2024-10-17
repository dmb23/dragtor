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
import pysbd

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


def _extract_new_propositions(sentence: str, existing_props: Propositions) -> Propositions:
    sys_prompt = """
    Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
    3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
    4. Present the results as a list of strings, formatted in JSON. Return only the JSON output.
    """


def get_propositions(text: str) -> Propositions:
    """
    - text -> sentences
    - sentence -> propositions
    - decide if proposition should be included or not
    """
    sys_prompt = """
    You recieve context and a relevant sentence. Decontextualize the sentence so that it is understandable on its own.
    For that add necessary modifiers to nouns or subphrases and replace pronouns (e.g. "it", "he", "she", "they", "this", "that") with the full name of the entities they refere to.
    Return only the decontextualized sentence.
    """

    user_prompt_template = """
    <context>
    {context}
    </context>

    Decontextualize the following sentence:
    <sentence>
    {sentence}
    </sentence>
    """

    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(text)
    logger.info(sentences)

    decontextualized = []
    for i, sentence in enumerate(sentences):
        context = " ".join(sentences[:i])

        messages = Messages()
        messages.system(sys_prompt)
        messages.user(user_prompt_template.format(context=context, sentence=sentence))

        answer = lsh.chat_llm(messages.format(), cache_prompt=True)
        logger.debug(f"---\n{sentence}\n-\n{answer}\n")

        decontextualized.append(answer)

    logger.info(decontextualized)

    return decontextualized


def get_propositions_old(text: str) -> Propositions:
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
    do_get_propositions = True
    do_get_faithfulness = False

    dl = data.JinaLoader()
    full_texts = dl.get_cache()
    logger.info(f"{len(full_texts)} cached texts available")

    contexts = {ident(text): text for text in full_texts}

    answers = json.loads(answers_file.read_text())
    text_key, topic_answers = next(iter(answers.items()))
    topic, questions = next(iter(topic_answers.items()))
    question, answer = next(iter(questions.items()))

    if do_get_propositions:
        props = get_propositions(answer)

        raise NotImplementedError
    else:
        props = Propositions(
            propositions=[
                "According to the document, weight training can help rock climbers improve their strength.",
                "Incorporating weight training into their training program can help climbers improve their climbing performance.",
            ]
        )

    logger.info(props)

    context = contexts[text_key]

    if do_get_faithfulness:
        eval_faithful = get_faithfullness(props, context)
    else:
        eval_faithful = EvalFaithful(
            evals=[
                Faithfulness(
                    statement="According to the document, weight training can help rock climbers improve their strength.",
                    reasoning="The document explicitly states that weight training can help rock climbers improve their strength, citing examples of specific exercises such as the deadlift that can help build strength without adding weight.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="Incorporating weight training into their training program can help climbers improve their climbing performance.",
                    reasoning="The document suggests that incorporating weight training into their training program can help climbers improve their climbing performance, highlighting the potential benefits of weight training for climbers.",
                    is_inferred=True,
                ),
            ]
        )

    logger.info(eval_faithful)

    model_answer = "The A2 pulley strain in the index finger is a common injury among climbers, but fortunately, it's often minor and can be treated effectively with conservative measures. Here's a step-by-step guide to help you recover:\n\n**Understanding the injury:**\nThe A2 pulley is a small, fibrous structure in the flexor tendon of the index finger, located near the base of the finger. It helps to stabilize the tendon and facilitate smooth movement of the finger. A strain or partial tear can occur due to repetitive stress, overuse, or a sudden jerk while climbing.\n\n**Conservative treatment:**\n\n1.  **Rest**: Give your finger a break and avoid climbing, heavy lifting, or activities that aggravate the pain. This allows the pulley to heal and reduces further irritation.\n2.  **Ice**: Apply ice to the affected area for 10-15 minutes, 2-3 times a day, to reduce inflammation and ease pain.\n3.  **Compression**: Use a compression bandage or wrap to provide support and stability to the finger.\n4.  **Elevation**: Elevate your hand above heart level to reduce swelling and promote blood flow.\n5.  **Pain management**: Over-the-counter pain relievers like acetaminophen or ibuprofen can help alleviate pain and reduce inflammation.\n\n**Stretching and strengthening exercises:**\n\n1.  **Finger bends**: Gently bend your index finger, keeping your wrist straight, and then release. Repeat for 10-15 repetitions, 3-4 times a day.\n2.  **Finger spreads**: Spread your index finger as far as you can, and then release. Repeat for 10-15 repetitions, 3-4 times a day.\n3.  **Finger extensions**: Hold a light weight (less than 1 pound) or a resistance band in your hand and slowly extend your index finger. Repeat for 10-15 repetitions, 3-4 times a day.\n4.  **Wrist rotations**: Hold a light weight or resistance band in your hand and rotate your wrist in both clockwise and counterclockwise directions. Repeat for 10-15 repetitions, 3-4 times a day.\n\n**Preventing future injuries:**\n\n1.  **Warm up**: Before climbing, warm up your hands and fingers with light exercises and stretching.\n2.  **Use proper climbing techniques**: Focus on using your legs and core to climb, rather than relying solely on your fingers.\n3.  **Take regular breaks**: Give your fingers a break and rest for a few minutes every hour to reduce fatigue and prevent overuse.\n4.  **Maintain finger health**: Regularly stretch and strengthen your fingers to prevent finger fatigue and reduce the risk of injury.\n\n**When to seek medical attention:**\nIf you experience any of the following symptoms, seek medical attention:\n\n*   Severe pain or swelling\n*   Difficulty moving your finger or wrist\n*   Numbness or tingling in your finger or hand\n*   A popping or snapping sound at the time of injury\n\nIf you're unsure about the severity of your injury or if you've experienced a more severe strain, consult with a medical professional or a sports medicine specialist for proper evaluation and treatment."

    gold_answer = "Rehabilitating an A2 pulley injury requires a comprehensive approach that addresses the injury's severity, promotes healing, and prevents future occurrences. Here are possible steps to rehab an A2 pulley injury:\n\n1.  **Initial Treatment**: The first step is to protect the injured area and avoid activities that exacerbate the injury. This may involve immobilizing the finger, using a splint or tape to support the pulley, and avoiding heavy lifting or bending.\n2.  **Active Range of Motion**: Once the initial inflammation has subsided, it's essential to start moving the finger through its range of motion. This can be done by gently flexing and extending the finger, and gradually increasing the range of motion over time.\n3.  **Soft Tissue Mobilization**: Soft tissue mobilization, such as instrument-assisted soft tissue mobilization (IASTM), can help promote healing and reduce scar tissue. This involves using a tool to apply gentle pressure to the affected area, promoting the breakdown of scar tissue and promoting healing.\n4.  **Diet and Sleep**: Adequate nutrition and sleep are crucial for healing. A diet rich in anti-inflammatory foods, such as salmon, and adequate sleep can help promote healing and reduce inflammation.\n5.  **Exercise**: Gentle exercises, such as finger bends and extensions, can help promote healing and strengthen the surrounding muscles. It's essential to avoid heavy lifting or bending, which can exacerbate the injury.\n6.  **H-Taping**: H-taping can provide additional support and stability to the injured area, helping to prevent further injury.\n7.  **Retraining**: Once the injury has healed, it's essential to retrain the finger to handle the stresses of climbing. This can involve gradually increasing the intensity and duration of climbing activities, as well as incorporating exercises to strengthen the surrounding muscles.\n8.  **Progressive V-Scale Ladder**: When returning to climbing, it's essential to follow a progressive V-scale ladder to avoid overexertion and prevent further injury. This involves gradually increasing the difficulty of climbs over time, allowing the finger to adapt to the stresses of climbing.\n\nIt's essential to note that the rehabilitation process may vary depending on the severity of the injury and individual factors. It's always best to consult with a medical professional or a qualified healthcare provider for personalized guidance and treatment."
