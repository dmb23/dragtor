import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    import json
    from typing import Annotated
    from typing_extensions import TypedDict

    from dragtor.config import conf
    import marimo as mo

    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    # from langgraph.prebuilt import ToolNode, tools_condition

    from langchain_groq.chat_models import ChatGroq
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import ToolMessage, BaseMessage

    os.environ["GROQ_API_KEY"] = conf.creds.groq
    os.environ["TAVILY_API_KEY"] = conf.creds.tavily
    return (
        Annotated,
        BaseMessage,
        ChatGroq,
        END,
        START,
        StateGraph,
        TavilySearchResults,
        ToolMessage,
        TypedDict,
        add_messages,
        conf,
        json,
        mo,
        os,
    )


@app.cell
def __(ChatGroq):
    model_name = "llama-3.1-70b-versatile"
    model = ChatGroq(model=model_name)
    return model, model_name


@app.cell(disabled=True)
def __(model):
    model.invoke("Hello, how are you?")
    return


@app.cell
def __(Annotated, TypedDict, add_messages):
    class State(TypedDict):
        # Messages have the type "list". The `add_messages` function
        # in the annotation defines how this state key should be updated
        # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]
    return (State,)


@app.cell
def __(State, model):
    def chatbot(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    return (chatbot,)


@app.cell
def __(END, START, State, StateGraph, chatbot):
    graph_builder = StateGraph(State)


    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()
    return graph, graph_builder


@app.cell
def __(graph, mo):
    mo.mermaid(graph.get_graph().draw_mermaid())
    return


@app.cell
def __(graph):
    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
    return (stream_graph_updates,)


@app.cell
def __(stream_graph_updates):
    question = "What can you tell me about finger training for climbing?"
    print(f"User: {question}")
    stream_graph_updates(question)

    if False:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                stream_graph_updates(user_input)
            except:
                # fallback if input() is not available
                user_input = "What do you know about LangGraph?"
                print("User: " + user_input)
                stream_graph_updates(user_input)
                break
    return question, user_input


@app.cell
def __(TavilySearchResults):
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    return tool, tools


@app.cell
def __(tool):
    tool.invoke("How often can I perform \"no hangs\" when climbing 3 times a week?")
    return


@app.cell
def __(State, model, tools):
    # Modification: tell the LLM which tools it can call
    model_with_tools = model.bind_tools(tools)

    def chatbot_w_tools(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    return chatbot_w_tools, model_with_tools


@app.cell
def __(ToolMessage, json):
    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""

        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}
    return (BasicToolNode,)


@app.cell
def __(END, State):
    def route_tools(
        state: State,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END
    return (route_tools,)


@app.cell
def __(
    BasicToolNode,
    END,
    START,
    State,
    StateGraph,
    chatbot_w_tools,
    route_tools,
    tool,
):
    gb2 = StateGraph(State)
    gb2.add_node("chatbot", chatbot_w_tools)

    tool_node = BasicToolNode(tools=[tool])
    gb2.add_node("tools", tool_node)


    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    gb2.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    gb2.add_edge("tools", "chatbot")
    gb2.add_edge(START, "chatbot")
    graph2 = gb2.compile()
    return gb2, graph2, tool_node


@app.cell
def __(graph2):
    def stream_graph_updates2(user_input: str):
        for event in graph2.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    question2 = "How often can I perform \"no hangs\" when climbing 3 times a week?"
    print(f"User: {question2}")
    stream_graph_updates2(question2)
    return question2, stream_graph_updates2


@app.cell
def __(graph2):
    graph2.get_state
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
