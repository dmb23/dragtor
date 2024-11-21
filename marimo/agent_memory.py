import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    from typing import Annotated

    import marimo as mo
    from dragtor.config import conf

    from langchain_groq import ChatGroq
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import BaseMessage
    from typing_extensions import TypedDict

    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver

    os.environ["GROQ_API_KEY"] = conf.creds.groq
    os.environ["TAVILY_API_KEY"] = conf.creds.tavily
    return (
        Annotated,
        BaseMessage,
        ChatGroq,
        END,
        MemorySaver,
        StateGraph,
        TavilySearchResults,
        ToolNode,
        TypedDict,
        add_messages,
        conf,
        mo,
        os,
        tools_condition,
    )


@app.cell
def __(Annotated, TypedDict, add_messages):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    return (State,)


@app.cell
def __(ChatGroq):
    model_name = "llama-3.1-70b-versatile"
    model = ChatGroq(model=model_name)
    return model, model_name


@app.cell
def __(State, model):
    def chatbot(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    return (chatbot,)


@app.cell
def __(END, State, StateGraph, chatbot):
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_edge("chatbot", END)
    return (graph_builder,)


@app.cell
def __(MemorySaver):
    memory = MemorySaver()
    return (memory,)


@app.cell
def __(graph_builder, memory, mo):
    graph = graph_builder.compile(checkpointer=memory)

    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell
def __():
    config = {"configurable": {"thread_id": "1"}}
    return (config,)


@app.cell
def __(config, graph):
    questions = ["How should I train finger strength for climbing?", "What sort of a health professional should I consult for these issues?"]

    for q in questions:
        events = graph.stream(
            {"messages": [("user", q)]},
            config,
            stream_mode="values"
        )
        for event in events:
            event["messages"][-1].pretty_print()
    return event, events, q, questions


@app.cell
def __(config, graph):
    snapshot = graph.get_state(config)
    print(snapshot.values["messages"])
    snapshot
    return (snapshot,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
