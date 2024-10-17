"""What should happen?

- write functionality to get propositions from an answer in a list
- get propositions from an answer
- check how many of those are grounded in the context

Also interesting:
- get propositions from a reference answer
- check how many TP / FP / FN I get?
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from dragtor import data, llm

lsh = llm.LlamaServerHandler.from_config()
logger.info(f"---\nModel:\n{lsh.modelpath}")

root_dir = Path("/Users/mischa/Projects/local/dragtor/")


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
        messages.format(),
        cache_prompt=True,
        json_schema=Propositions.model_json_schema(),
        n_predict=2048,
    )

    propositions = Propositions.model_validate_json(answer)

    return propositions


class Faithfulness(BaseModel):
    statement: str
    reasoning: str
    is_inferred: bool


class EvalFaithful(BaseModel):
    evals: list[Faithfulness]

    def fraction_true(self) -> float:
        return len([e for e in self.evals if e.is_inferred]) / len(self.evals)


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


class ClassificationCategory(Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"


class TruthCategory(BaseModel):
    statement: str
    reasoning: str
    classification: ClassificationCategory


class AnswerCorrectness(BaseModel):
    statements: list[TruthCategory]


class AnswerCorrectnessEval(BaseModel):
    tp: list[TruthCategory]
    fp: list[TruthCategory]
    fn: list[TruthCategory]

    def f1_score(self):
        if len(self.tp) == 0:
            return 0
        return len(self.tp) / (len(self.tp) + 0.5 * (len(self.fp) + len(self.fn)))


if __name__ == "__main__":
    do_get_propositions = False
    do_get_faithfulness = False
    do_get_answer_correctness = False

    dl = data.JinaLoader()
    full_texts = dl.get_cache()
    logger.info(f"{len(full_texts)} cached texts available")

    contexts = {ident(text): text for text in full_texts}

    answers_file = root_dir / "data" / "features" / "01_answers.json"
    text_key = "091546bddf04cb106b916a4843ab7e90"
    topic = "Treatment and Rehabilitation of A2 Pulley Injuries"
    question = "Can you provide specific examples of exercises and retraining protocols that are recommended for A2 pulley injuries, particularly in the early stages of rehabilitation?"

    answers = json.loads(answers_file.read_text())
    answer = answers[text_key][topic][question]
    # topic_answers = next(iter(answers.items()))
    # topic, questions = next(iter(topic_answers.items()))
    # question, answer = next(iter(questions.items()))

    logger.info(f"Question: {question}\nAnswer: {answer}")

    if do_get_propositions:
        props = get_propositions(answer)
    else:
        props = Propositions(
            propositions=[
                "The document provides specific examples of exercises and retraining protocols for A2 pulley injuries in the early stages of rehabilitation.",
                "Initial levels of retraining include Level 1: Palm crimps with 1-3 second holds and relaxation.",
                "Initial levels of retraining include Level 2: Puddy crimps with 4-5 minutes of holds and breaks.",
                "Initial levels of retraining include Level 3: Farmer crimps with progressive weight, starting with 5 pounds and increasing as tolerated.",
                "Tissue loading should be done slowly and safely.",
                "Tissue loading should trigger the body to strengthen and support the injured area.",
                "Eccentric tissue loading is mentioned, but specific exercises are not provided in the document.",
                "The document emphasizes the importance of listening to the body and stopping if any pain or discomfort is experienced during retraining.",
                "It's recommended to seek professional help if unsure about the severity of the injury or the best course of treatment.",
                "Gentle, low-intensity exercises promote tissue loading and adaptation in the early stages of rehabilitation.",
                "More intense and specific exercises can be introduced as the injury progresses.",
                "A gradual and controlled progression is crucial to avoid reinjury.",
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
                    statement="The document provides specific examples of exercises and retraining protocols for A2 pulley injuries in the early stages of rehabilitation.",
                    reasoning="The document specifically mentions Level 1: Palm crimps, Level 2: Puddy Crimps, and Level 3: Farmer Crimps as initial levels of retraining. It also discusses the importance of progressive weight and tissue loading. These protocols suggest that the document provides specific examples of exercises and retraining protocols for A2 pulley injuries in the early stages of rehabilitation.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="Initial levels of retraining include Level 1: Palm crimps with 1-3 second holds and relaxation.",
                    reasoning="The document explicitly states Level 1: Palm crimps as an initial level of retraining, along with specific guidelines for duration and relaxation.",
                    is_inferred=False,
                ),
                Faithfulness(
                    statement="Initial levels of retraining include Level 2: Puddy crimps with 4-5 minutes of holds and breaks.",
                    reasoning="The document explicitly states Level 2: Puddy crimps as an initial level of retraining, along with specific guidelines for duration and breaks.",
                    is_inferred=False,
                ),
                Faithfulness(
                    statement="Initial levels of retraining include Level 3: Farmer crimps with progressive weight, starting with 5 pounds and increasing as tolerated.",
                    reasoning="The document explicitly states Level 3: Farmer Crimps as an initial level of retraining, along with specific guidelines for progressive weight and tolerance.",
                    is_inferred=False,
                ),
                Faithfulness(
                    statement="Tissue loading should be done slowly and safely.",
                    reasoning="The document emphasizes the importance of slow and safe tissue loading, suggesting that this is a recommended approach.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="Tissue loading should trigger the body to strengthen and support the injured area.",
                    reasoning="The document explicitly states that tissue loading should trigger the body to strengthen and support the injured area, suggesting that this is a key goal of retraining.",
                    is_inferred=False,
                ),
                Faithfulness(
                    statement="Eccentric tissue loading is mentioned, but specific exercises are not provided in the document.",
                    reasoning="The document mentions eccentric tissue loading, but does not provide specific exercises, suggesting that this is a topic that requires further research or consultation with a medical professional.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="The document emphasizes the importance of listening to the body and stopping if any pain or discomfort is experienced during retraining.",
                    reasoning="The document explicitly states that it is essential to listen to the body and stop if any pain or discomfort is experienced during retraining, suggesting that this is a key principle of safe and effective retraining.",
                    is_inferred=False,
                ),
                Faithfulness(
                    statement="It's recommended to seek professional help if unsure about the severity of the injury or the best course of treatment.",
                    reasoning="The document mentions seeking professional help if unsure about the severity of the injury or the best course of treatment, suggesting that this is a recommended approach.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="Gentle, low-intensity exercises promote tissue loading and adaptation in the early stages of rehabilitation.",
                    reasoning="The document suggests that gentle, low-intensity exercises are used in the early stages of rehabilitation to promote tissue loading and adaptation, which is consistent with the principles of tissue adaptation and regeneration.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="More intense and specific exercises can be introduced as the injury progresses.",
                    reasoning="The document suggests that more intense and specific exercises can be introduced as the injury progresses, which is consistent with the principles of progressive overload and specific training.",
                    is_inferred=True,
                ),
                Faithfulness(
                    statement="A gradual and controlled progression is crucial to avoid reinjury.",
                    reasoning="The document emphasizes the importance of gradual and controlled progression to avoid reinjury, which is a key principle of safe and effective retraining.",
                    is_inferred=True,
                ),
            ]
        )
    logger.info(eval_faithful)
    logger.info(f"{eval_faithful.fraction_true()=}")

    if do_get_answer_correctness:
        model_answer = "The A2 pulley strain in the index finger is a common injury among climbers, but fortunately, it's often minor and can be treated effectively with conservative measures. Here's a step-by-step guide to help you recover:\n\n**Understanding the injury:**\nThe A2 pulley is a small, fibrous structure in the flexor tendon of the index finger, located near the base of the finger. It helps to stabilize the tendon and facilitate smooth movement of the finger. A strain or partial tear can occur due to repetitive stress, overuse, or a sudden jerk while climbing.\n\n**Conservative treatment:**\n\n1.  **Rest**: Give your finger a break and avoid climbing, heavy lifting, or activities that aggravate the pain. This allows the pulley to heal and reduces further irritation.\n2.  **Ice**: Apply ice to the affected area for 10-15 minutes, 2-3 times a day, to reduce inflammation and ease pain.\n3.  **Compression**: Use a compression bandage or wrap to provide support and stability to the finger.\n4.  **Elevation**: Elevate your hand above heart level to reduce swelling and promote blood flow.\n5.  **Pain management**: Over-the-counter pain relievers like acetaminophen or ibuprofen can help alleviate pain and reduce inflammation.\n\n**Stretching and strengthening exercises:**\n\n1.  **Finger bends**: Gently bend your index finger, keeping your wrist straight, and then release. Repeat for 10-15 repetitions, 3-4 times a day.\n2.  **Finger spreads**: Spread your index finger as far as you can, and then release. Repeat for 10-15 repetitions, 3-4 times a day.\n3.  **Finger extensions**: Hold a light weight (less than 1 pound) or a resistance band in your hand and slowly extend your index finger. Repeat for 10-15 repetitions, 3-4 times a day.\n4.  **Wrist rotations**: Hold a light weight or resistance band in your hand and rotate your wrist in both clockwise and counterclockwise directions. Repeat for 10-15 repetitions, 3-4 times a day.\n\n**Preventing future injuries:**\n\n1.  **Warm up**: Before climbing, warm up your hands and fingers with light exercises and stretching.\n2.  **Use proper climbing techniques**: Focus on using your legs and core to climb, rather than relying solely on your fingers.\n3.  **Take regular breaks**: Give your fingers a break and rest for a few minutes every hour to reduce fatigue and prevent overuse.\n4.  **Maintain finger health**: Regularly stretch and strengthen your fingers to prevent finger fatigue and reduce the risk of injury.\n\n**When to seek medical attention:**\nIf you experience any of the following symptoms, seek medical attention:\n\n*   Severe pain or swelling\n*   Difficulty moving your finger or wrist\n*   Numbness or tingling in your finger or hand\n*   A popping or snapping sound at the time of injury\n\nIf you're unsure about the severity of your injury or if you've experienced a more severe strain, consult with a medical professional or a sports medicine specialist for proper evaluation and treatment."

        gold_answer = "Rehabilitating an A2 pulley injury requires a comprehensive approach that addresses the injury's severity, promotes healing, and prevents future occurrences. Here are possible steps to rehab an A2 pulley injury:\n\n1.  **Initial Treatment**: The first step is to protect the injured area and avoid activities that exacerbate the injury. This may involve immobilizing the finger, using a splint or tape to support the pulley, and avoiding heavy lifting or bending.\n2.  **Active Range of Motion**: Once the initial inflammation has subsided, it's essential to start moving the finger through its range of motion. This can be done by gently flexing and extending the finger, and gradually increasing the range of motion over time.\n3.  **Soft Tissue Mobilization**: Soft tissue mobilization, such as instrument-assisted soft tissue mobilization (IASTM), can help promote healing and reduce scar tissue. This involves using a tool to apply gentle pressure to the affected area, promoting the breakdown of scar tissue and promoting healing.\n4.  **Diet and Sleep**: Adequate nutrition and sleep are crucial for healing. A diet rich in anti-inflammatory foods, such as salmon, and adequate sleep can help promote healing and reduce inflammation.\n5.  **Exercise**: Gentle exercises, such as finger bends and extensions, can help promote healing and strengthen the surrounding muscles. It's essential to avoid heavy lifting or bending, which can exacerbate the injury.\n6.  **H-Taping**: H-taping can provide additional support and stability to the injured area, helping to prevent further injury.\n7.  **Retraining**: Once the injury has healed, it's essential to retrain the finger to handle the stresses of climbing. This can involve gradually increasing the intensity and duration of climbing activities, as well as incorporating exercises to strengthen the surrounding muscles.\n8.  **Progressive V-Scale Ladder**: When returning to climbing, it's essential to follow a progressive V-scale ladder to avoid overexertion and prevent further injury. This involves gradually increasing the difficulty of climbs over time, allowing the finger to adapt to the stresses of climbing.\n\nIt's essential to note that the rehabilitation process may vary depending on the severity of the injury and individual factors. It's always best to consult with a medical professional or a qualified healthcare provider for personalized guidance and treatment."

        props_model_answer = get_propositions(model_answer)
        logger.debug(f"{props_model_answer=}")
        props_gold_answer = get_propositions(gold_answer)
        logger.debug(f"{props_gold_answer=}")

        sys_prompt = """
        Given a ground truth and statements from an answer, analyze each statement and classify them in one of the following categories:
        TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth. Also statements that are present in the ground truth and have an equivalent statement in the answer.
        FP (false positive): statements present in the answer but not directly supported by any statement in ground truth.
        FN (false negative): statements found in the ground truth but not present in answer.
        Each statement can only belong to one of the categories.
        Provide a reason for each classification.
        Output a JSON array with objects for each statement, containing the statement, the reasoning and the classification:
        [
            {
                "statement": <text>,
                "reason": <text>,
                "classification": <text>
            },
            ...
        ]
        """

        user_prompt_template = """
        Classify the following statements:

        <ground truth>
        {ground_truth}
        </ground truth>

        <answer>
        {answer}
        </answer>
        """

        messages = Messages()
        messages.system(sys_prompt)
        user_message = user_prompt_template.format(
            ground_truth=props_gold_answer, answer=props_model_answer
        )
        logger.debug(f"{user_message=}")
        messages.user(user_message)

        res = lsh.chat_llm(
            messages.format(),
            cache_prompt=True,
            json_schema=AnswerCorrectness.model_json_schema(),
            n_predict=4096,
        )
        answer_correctness = AnswerCorrectness.model_validate_json(res)

        logger.info(answer_correctness)

        eval = AnswerCorrectnessEval(
            tp=[s for s in answer_correctness.statements if s.classification.value == "TP"],
            fp=[s for s in answer_correctness.statements if s.classification.value == "FP"],
            fn=[s for s in answer_correctness.statements if s.classification.value == "FN"],
        )

    else:
        eval = AnswerCorrectnessEval(
            tp=[
                TruthCategory(
                    statement="Initial treatment for an A2 pulley injury involves protecting the injured area and avoiding activities that exacerbate the injury.",
                    reasoning="Found in the answer, but with different specifics. The answer mentions protecting the injured area, avoiding heavy lifting or bending, and using a splint or tape to support the pulley.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Immobilizing the finger and using a splint or tape to support the pulley are possible initial treatment methods.",
                    reasoning="Found in the answer with similar specifics. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Avoiding heavy lifting or bending is essential during initial treatment.",
                    reasoning="Found in the answer with similar specifics. The answer mentions avoiding heavy lifting or bending to prevent exacerbating the injury.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Active range of motion is essential once initial inflammation has subsided.",
                    reasoning="Found in the answer, but with different specifics. The answer mentions gradually increasing the range of motion over time and performing finger bends, spreads, and extensions to recover from the A2 pulley strain.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Gradually increasing the range of motion over time is crucial for active range of motion.",
                    reasoning="Found in the answer with similar specifics. The answer mentions gradually increasing the range of motion over time and performing gentle exercises to promote healing.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Adequate nutrition and sleep are crucial for healing.",
                    reasoning="Found in the answer with similar specifics. The answer mentions the importance of proper nutrition and rest for healing.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="A diet rich in anti-inflammatory foods, such as salmon, can help promote healing and reduce inflammation.",
                    reasoning="Found in the answer with similar specifics. The answer mentions the importance of proper nutrition and rest for healing.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Adequate sleep is essential for healing and reducing inflammation.",
                    reasoning="Found in the answer with similar specifics. The answer mentions the importance of proper nutrition and rest for healing.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Gentle exercises, such as finger bends and extensions, can help promote healing and strengthen surrounding muscles.",
                    reasoning="Found in the answer with similar specifics. The answer mentions finger bends, spreads, and extensions as recovery methods.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Avoiding heavy lifting or bending is essential to prevent exacerbating the injury.",
                    reasoning="Found in the answer with similar specifics. The answer mentions avoiding heavy lifting or bending to prevent exacerbating the injury.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Retraining the finger is essential once the injury has healed.",
                    reasoning="Found in the answer with similar specifics. The answer mentions retraining the finger as part of the recovery process.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Retraining involves gradually increasing the intensity and duration of climbing activities.",
                    reasoning="Found in the answer with similar specifics. The answer mentions gradually increasing the difficulty of climbs over time as part of retraining.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Exercises to strengthen surrounding muscles are necessary for retraining.",
                    reasoning="Found in the answer with similar specifics. The answer mentions performing gentle exercises to recover from the A2 pulley strain.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Following a progressive V-scale ladder is essential when returning to climbing.",
                    reasoning="Found in the answer with similar specifics. The answer mentions gradually increasing the difficulty of climbs over time as part of the progressive V-scale ladder.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="A progressive V-scale ladder helps avoid overexertion and prevents further injury.",
                    reasoning="Found in the answer with similar specifics. The answer mentions avoiding overexertion and preventing future injuries as part of the progressive V-scale ladder.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Gradually increasing the difficulty of climbs over time is necessary for a progressive V-scale ladder.",
                    reasoning="Found in the answer with similar specifics. The answer mentions gradually increasing the difficulty of climbs over time as part of the progressive V-scale ladder.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="The rehabilitation process may vary depending on the severity of the injury and individual factors.",
                    reasoning="Found in the answer with similar specifics. The answer mentions the importance of consulting with a medical professional for personalized guidance and treatment.",
                    classification=ClassificationCategory.TP,
                ),
                TruthCategory(
                    statement="Consulting with a medical professional or a qualified healthcare provider is essential for personalized guidance and treatment.",
                    reasoning="Found in the answer with similar specifics. The answer mentions consulting with a medical professional or a sports medicine specialist for proper evaluation and treatment.",
                    classification=ClassificationCategory.TP,
                ),
            ],
            fp=[],
            fn=[
                TruthCategory(
                    statement="Rehabilitating an A2 pulley injury requires a comprehensive approach.",
                    reasoning="Not found in the answer. The answer discusses specific treatments and recovery methods, but does not mention a comprehensive approach.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="A comprehensive approach to rehabilitating an A2 pulley injury addresses the injury's severity, promotes healing, and prevents future occurrences.",
                    reasoning="Not found in the answer. The answer discusses specific treatments and recovery methods, but does not mention a comprehensive approach or prevention of future occurrences.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="Gently flexing and extending the finger can help promote active range of motion.",
                    reasoning="Not found in the answer. The answer mentions finger bends, spreads, and extensions, but does not specify gentle flexing and extending.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="Soft tissue mobilization, such as instrument-assisted soft tissue mobilization (IASTM), can help promote healing and reduce scar tissue.",
                    reasoning="Not found in the answer. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="Soft tissue mobilization involves using a tool to apply gentle pressure to the affected area.",
                    reasoning="Not found in the answer. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="Breakdown of scar tissue and promotion of healing are benefits of soft tissue mobilization.",
                    reasoning="Not found in the answer. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="H-taping can provide additional support and stability to the injured area.",
                    reasoning="Not found in the answer. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.FN,
                ),
                TruthCategory(
                    statement="H-taping helps prevent further injury.",
                    reasoning="Not found in the answer. The answer mentions using a compression bandage or wrap to provide support and stability to the finger.",
                    classification=ClassificationCategory.FN,
                ),
            ],
        )

    logger.info(eval)
    logger.info(f"{eval.f1_score()=}")
