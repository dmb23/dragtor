from datetime import datetime
from enum import Enum
from functools import cached_property
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, PrivateAttr, computed_field

from dragtor import config
from dragtor.llm import LlamaServerHandler
from dragtor.llm.dragtor import LocalDragtor
from dragtor.utils import Messages

# ruff: noqa: E501
# -> lots of long example strings

lsh = LlamaServerHandler.from_config()

# these magic numbers should be easy to adjust
# Ideal: automatically adjust to the length of the question / number of propositions
proposition_output_tokens = 2048
faithfulness_output_tokens = 2048
correctness_output_tokens = 4096


class Propositions(BaseModel):
    """Individual statements from a text, understandable in isolation"""

    propositions: list[str]


class _Faithfulness(BaseModel):
    """Is a single statement inferred from a given context"""

    statement: str
    reasoning: str
    is_inferred: bool


class EvalFaithful(BaseModel):
    """Evaluation of a RAG answer: which statements are inferred from the context?"""

    evals: list[_Faithfulness]

    def fraction_true(self) -> float:
        return len([e for e in self.evals if e.is_inferred]) / len(self.evals)


class ClassificationCategory(Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"


class TruthCategory(BaseModel):
    """Evaluate an individual statement from a comparison model answer against "gold-truth".

    True Positive: model answer statement is present in "gold-truth" answer
    False Positive: model answer statement is not present in "gold-truth" answer
    False Negative: "gold-truth" answer statement is not present in model answer
    """

    statement: str
    reasoning: str
    classification: ClassificationCategory


class AnswerCorrectness(BaseModel):
    """Evaluate all statements in a model answer against a "gold-truth" answer"""

    statements: list[TruthCategory]


class EvalAnswerCorrectness(BaseModel):
    """Evaluation of a RAG answer: which of the statements are also present in the gold-truth?"""

    tp: list[TruthCategory]
    fp: list[TruthCategory]
    fn: list[TruthCategory]

    def f1_score(self) -> float:
        if len(self.tp) == 0:
            return 0
        return len(self.tp) / (len(self.tp) + 0.5 * (len(self.fp) + len(self.fn)))


def _get_propositions(text: str) -> Propositions:
    """Extract propositions from a given text.

    I.e. extract all statements from a text in a way that they can be understood in isolation.

    prompt taken from [langchain](https://github.com/langchain-ai/langchain/blob/master/templates/propositional-retrieval/propositional_retrieval/proposal_chain.py)
    """
    logger.debug("Calculating Propositions from statement")
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

    answer = lsh.chat_llm(
        messages,
        cache_prompt=True,
        json_schema=Propositions.model_json_schema(),
        n_predict=proposition_output_tokens,
    )

    propositions = Propositions.model_validate_json(answer)

    return propositions


def _get_faithfullness(answer: str, context: str) -> EvalFaithful:
    """Which statements made in an answer are inferred from the context?

    Allows to measure how strongly an answer is only based on the provided context
    and how strongly the LLM includes background knowledge.

    Taken mostly from [Ragas](https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py).
    """
    logger.debug("Calculating Answer Faithfulness")
    props = _get_propositions(answer)

    def _format_user_message(statements: list[str], context: str) -> str:
        """format used for querying the model"""
        return json.dumps({"context": context, "statements": statements})

    sys_prompt = """
    Your task is to judge the faithfulness of a series of statements based on a given context.
    For each statement you must return is_inferred as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
    For each statement, return the statement, the reasoning and the value for is_inferred in JSON.
    Output only the final JSON array, nothing else.
    """

    # few shot examples
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
            _Faithfulness(
                statement="John is majoring in Biology.",
                reasoning="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                is_inferred=False,
            ),
            _Faithfulness(
                statement="John is taking a course on Artificial Intelligence.",
                reasoning="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                is_inferred=False,
            ),
            _Faithfulness(
                statement="John is a dedicated student.",
                reasoning="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                is_inferred=True,
            ),
            _Faithfulness(
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
        messages,
        cache_prompt=True,
        json_schema=EvalFaithful.model_json_schema(),
        n_predict=faithfulness_output_tokens,
    )

    ef = EvalFaithful.model_validate_json(answer)

    return ef


def _get_answer_correctness(model_answer: str, gold_answer: str) -> EvalAnswerCorrectness:
    """Evaluate a model answer against a "gold-truth" reference.

    Following mostly concepts from [Ragas](https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_answer_correctness.py)
    """
    logger.debug("Calculating Answer Correctness")
    props_model_answer = _get_propositions(model_answer)
    props_gold_answer = _get_propositions(gold_answer)

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
    messages.user(user_message)

    res = lsh.chat_llm(
        messages,
        cache_prompt=True,
        json_schema=AnswerCorrectness.model_json_schema(),
        n_predict=correctness_output_tokens,
    )
    answer_correctness = AnswerCorrectness.model_validate_json(res)

    eval = EvalAnswerCorrectness(
        tp=[s for s in answer_correctness.statements if s.classification.value == "TP"],
        fp=[s for s in answer_correctness.statements if s.classification.value == "FP"],
        fn=[s for s in answer_correctness.statements if s.classification.value == "FN"],
    )

    return eval


class QuestionEvaluator(BaseModel):
    question: str
    _dragtor: LocalDragtor = PrivateAttr(default_factory=LocalDragtor)

    @computed_field
    @cached_property
    def context(self) -> str:
        _context = self._dragtor._get_context(self.question)
        logger.info(f'Using context "{_context}"')
        return _context

    @computed_field
    @cached_property
    def answer(self) -> str:
        _answer = self._dragtor.chat(self.question)
        logger.info(f'Using answer "{_answer}"')
        return _answer

    @computed_field
    @property
    def gold_truth(self) -> str:
        # check for available gold_truth
        gold_truth_file = (
            Path(config.conf.base_path) / config.conf.eval.eval_dir / config.conf.eval.eval_answers
        )
        if gold_truth_file.exists() and gold_truth_file.is_file():
            gold_answers = json.loads(gold_truth_file.read_text())
            logger.info(f'Using gold truth answer "{gold_answers.get(self.question, "")}"')
            return gold_answers.get(self.question, "")
        return ""

    @computed_field
    @cached_property
    def faithfullness(self) -> EvalFaithful:
        logger.info(f'Calculating Faithfullness for question "{self.question}"')
        return _get_faithfullness(self.answer, self.context)

    @computed_field
    @cached_property
    def correctness(self) -> EvalAnswerCorrectness | None:
        if self.gold_truth:
            logger.info(f'Calculating Correctness for question "{self.question}"')
            return _get_answer_correctness(self.answer, self.gold_truth)
        return None

    def run_eval(self):
        logger.info("Evaluating how faithfull the answer is to the context")
        logger.info(f"Faithfulness: {self.faithfullness.fraction_true():.0%}")

        if self.correctness:
            logger.info('Evaluating how close the answer is to the "golden" reference')
            logger.info(f"Correctness: {self.correctness.f1_score():.0%}")


class EvaluationSuite(BaseModel):
    @computed_field
    @cached_property
    def gold_answers(self) -> dict[str, str]:
        gold_truth_file = (
            Path(config.conf.base_path) / config.conf.eval.eval_dir / config.conf.eval.eval_answers
        )
        if not gold_truth_file.is_file():
            logger.warning(
                f"Can't evaluate RAG performance: Expected reference questions and answers at {gold_truth_file}"
            )
        return json.loads(gold_truth_file.read_text())

    @computed_field
    @cached_property
    def questions(self) -> list[str]:
        return list(self.gold_answers.keys())

    @computed_field
    @cached_property
    def evaluations(self) -> dict[str, QuestionEvaluator]:
        _dragtor = LocalDragtor()
        return {
            question: QuestionEvaluator(question=question, _dragtor=_dragtor)
            for question in self.questions
        }

    def run_all_evals(self):
        logger.info(f"Evaluating {len(self.questions)} questions")
        for i, question in enumerate(self.questions):
            logger.info(f"Evaluating Question {i+1}: {question}")
            self.evaluations[question].run_eval()

        eval_file = (
            Path(config.conf.base_path) / config.conf.eval.eval_dir / f"{datetime.now()}_eval.json"
        )
        eval_file.write_text(self.model_dump_json())
        logger.info("Final Evaluations:")
        for i, question in enumerate(self.questions):
            logger.info(f"Question {i+1}:")
            logger.info(
                f"Faithfulness: {self.evaluations[question].faithfullness.fraction_true():.0%}"
            )
            if self.evaluations[question].correctness:
                logger.info(f"Correctness: {self.evaluations[question].correctness.f1_score():.2f}")

    @classmethod
    def summarize_dump(cls, filepath: Path):
        res = json.loads(filepath.read_text())
        # TODO: re-calculate metrics from dumped values
