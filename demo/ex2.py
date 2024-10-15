from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

_RED = "\033[91m"
_GREEN = "\033[92m"
_ENDC = "\033[0m"


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o")


def chatbot(state: State):
    # return the modifier for the state
    return {"messages": [llm.invoke(state["messages"])]}


def user(state: State):
    return {"messages": [HumanMessage(content=input(f"{_GREEN}User: {_ENDC}"))]}


# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("user", user)

graph_builder.add_edge(START, "user")
graph_builder.add_edge("chatbot", "user")
# add router
graph_builder.add_conditional_edges("user", lambda state: "chatbot" if state["messages"][-1].content != "quit" else END)


# Compile & run
graph = graph_builder.compile()
for event in graph.stream({"messages": []}):
    for value in event.values():
        if not isinstance(value["messages"][-1], HumanMessage):
            print(f"{_RED}Assistant:{_ENDC}", value["messages"][-1].content)
