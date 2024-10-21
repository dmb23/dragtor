from dataclasses import dataclass

from pydantic import BaseModel


class StatementTruth(BaseModel):
    statement: str
    reasoning: str
    is_inferred: bool


@dataclass
class EvalResult:
    question: str
    context: str
    answer: str
    statements: list[StatementTruth]
    gold_truth: str | None


@dataclass
class EvalSuiteResult:
    evals: list[EvalResult]
    scores: dict


class Evaluator:
    def evaluate_answer(self, question: str) -> EvalResult:
        """
        - get the context for a question
        - answer the question based on the context
        - evaluate the answer for truthfullness, i.e.
            - extract propositions
            - check which of those are based on the context
        - search for a gold-truth answer
            - if present, evaluate against that answer
        """
        raise NotImplementedError

    def evaluate_gold_truths(self) -> EvalSuiteResult:
        """
        - load all available gold-truth QA pairs
        - evaluate each question individually, and against the gold-truth
        - calculate scores from multiple results
        """
        raise NotImplementedError
