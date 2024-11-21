import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    from typing import Annotated

    import marimo as mo
    from pydantic import BaseModel
    from dragtor.config import conf

    from langchain_groq import ChatGroq
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
    from typing_extensions import TypedDict

    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver

    os.environ["GROQ_API_KEY"] = conf.creds.groq
    os.environ["TAVILY_API_KEY"] = conf.creds.tavily
    return (
        AIMessage,
        Annotated,
        BaseMessage,
        BaseModel,
        ChatGroq,
        END,
        HumanMessage,
        MemorySaver,
        StateGraph,
        SystemMessage,
        TavilySearchResults,
        ToolMessage,
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
        ask_human: bool
    return (State,)


@app.cell
def __(ChatGroq):
    model_name = "llama-3.1-70b-versatile"
    model = ChatGroq(model=model_name)
    return model, model_name


@app.cell
def __(BaseModel, TavilySearchResults, model):
    class RequestAssistance(BaseModel):
        """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

        To use this function, relay the user's 'request' so the expert can provide the right guidance.
        """
        request: str

    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool]
    model_with_tools = model.bind_tools(tools + [RequestAssistance])
    return RequestAssistance, model_with_tools, search_tool, tools


@app.cell
def __(RequestAssistance, State, model_with_tools):
    def chatbot(state: State):
        response = model_with_tools.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
            
        return {"messages": [response], "ask_human": ask_human}
    return (chatbot,)


@app.cell
def __(AIMessage, State, ToolMessage):
    def create_response(response: str, ai_message: AIMessage):
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0]["id"]
        )

    def human_node(state: State):
        new_messages = []
        if not isinstance(state["messages"][-1], ToolMessage):
            # Typically, the user will have updated the state during the interrupt.
            # IF they choose not to, we will include a placeholder Toolmessage
            # to let the LLM continue.
            new_messages.append(create_response("No response from human.", state["messages"][-1]))
        return {
            # append the new messages
            "messages": new_messages,
            # unset the flag
            "ask_human": False
        }
    return create_response, human_node


@app.cell
def __(State, tools_condition):
    def select_next_node(state: State):
        if state["ask_human"]:
            return "human"
        return tools_condition(state)
    return (select_next_node,)


@app.cell
def __(
    END,
    State,
    StateGraph,
    ToolNode,
    chatbot,
    human_node,
    select_next_node,
    tools,
):
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_node("human", human_node)

    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")

    return (graph_builder,)


@app.cell
def __(MemorySaver):
    memory = MemorySaver()
    return (memory,)


@app.cell
def __(graph_builder, memory, mo):
    graph = graph_builder.compile(
        checkpointer=memory,interrupt_before=["human"]
    )

    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell
def __():
    config = {"configurable": {"thread_id": "1"}}
    return (config,)


@app.cell
def __(config, graph):
    system_prompt = "You are an helpful assistant who always thinks first what should be the next step of action."
    question_for_human = "I need some support for discussing my training plan. Could you request assistance for me?"

    events = graph.stream(
        {"messages": [
            ("system", system_prompt),
            ("user", question_for_human),
        ]},
        config,
        stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
        else:
            print(event)
    return event, events, question_for_human, system_prompt


@app.cell
def __(config, graph):
    snapshot = graph.get_state(config)
    print(snapshot.next)
    print(snapshot.values["messages"][0])
    print(snapshot.values["messages"][-1])
    return (snapshot,)


@app.cell
def __(config, create_response, graph, snapshot):
    # manually adjust the state
    ai_message = snapshot.values["messages"][-1]
    human_response = (
        "I, as an expert, say your training plan looks good."
    )
    tool_message = create_response(human_response, ai_message)
    graph.update_state(config, {"messages": [tool_message]})

    graph.get_state(config).values["messages"]
    return ai_message, human_response, tool_message


@app.cell
def __(config, graph):
    graph.get_state(config).next
    return


@app.cell
def __(config, graph):
    # resume the graph
    _events = graph.stream(None, config, stream_mode="values")
    for _event in _events:
        if "messages" in _event:
            _event["messages"][-1].pretty_print()
        else:
            print(_event)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
