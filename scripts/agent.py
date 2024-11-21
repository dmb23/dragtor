"""Use Groq Llama 70B for Agent applications.

Write some tools to use and play around with stuff?
"""

from enum import Enum
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import END
from loguru import logger
from pydantic import BaseModel

from dragtor.config import conf
from dragtor.index.index import get_index
from dragtor.llm.dragtor import LocalDragtor
from dragtor.utils import Messages

os.environ["GROQ_API_KEY"] = conf.creds.groq

"""PLAN:

    - start
    - question
    - query index
    - check value
    -   - END?
    - query full text
    -   - END?
"""

model_name = "llama-3.1-70b-versatile"
model = ChatGroq(model=model_name)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list
    answer: str
    loop_step: int


def retrieve_documents(state: State) -> dict:
    """node function: works on state"""
    logger.info("Retrieving documents")
    index = get_index()
    question = state["messages"][-1].content
    results = index.query(question)

    return {"question": question, "documents": results}


def generate_answer(state: State) -> dict:
    """node function: works on state"""
    logger.info("Generating Answer")
    dragtor = LocalDragtor()

    loop_step = state.get("loop_step", 0)

    question = state["question"]
    context = "\n\n".join(state["documents"])
    messages = dragtor._to_messages(question, context)
    answer = dragtor.llm.chat_llm(messages)

    return {"answer": answer, "loop_step": loop_step + 1}


def check_answer(state: State) -> str:
    """edge function: only returns a string"""
    if state.get("loop_step", 0) >= 2:
        logger.info("Max retries reached - stopping")
        return "max_retries"

    system_prompt = """
    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of factual means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of wrong means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    user_prompt_template = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'factual' or 'wrong' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    class Score(Enum):
        FACTUAL = "factual"
        WRONG = "wrong"

    class Grading(BaseModel):
        binary_score: Score
        explanation: str

    messages = Messages()
    messages.system(system_prompt)
    messages.user(
        user_prompt_template.format(documents=state["documents"], generation=state["answer"])
    )

    grading_model = model.with_structured_output(Grading)
    response = grading_model.invoke(messages.format())

    logger.debug(response)

    grading = Grading.model_validate(response)

    if grading.binary_score == Score.FACTUAL:
        logger.info(f"Factual answer: {grading.explanation}")
    else:
        logger.info(f"Wrong answer: {grading.explanation}")

    return grading.binary_score.name


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    graph_builder.add_node("retrieve_documents", retrieve_documents)
    graph_builder.add_node("generate_answer", generate_answer)
    # TODO: graph_builder.add_node("answer_message", answer_message)

    graph_builder.set_entry_point("retrieve_documents")
    graph_builder.add_edge("retrieve_documents", "generate_answer")
    graph_builder.add_conditional_edges(
        "generate_answer",
        check_answer,
        {"max_retries": END, "FACTUAL": END, "WRONG": "generate_answer"},
    )

    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    question = "How can I treat a mild strain of the A2 pulley?"
    inputs = {"messages": [HumanMessage(content=question)]}

    events = app.stream(inputs, config, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()
