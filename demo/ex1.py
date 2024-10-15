from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    # var: Annotated[type, reducer]
    # reducer is a function that takes the current value and the new value and returns the new value
    messages: Annotated[list, add_messages]  # append new messages to the list


llm = ChatOpenAI(model="gpt-4o")


def chatbot(state: State):
    # return the modifier for the state
    return {"messages": [llm.invoke(state["messages"])]}


# Build graph
graph_builder = StateGraph(State)
# register the chatbot node
graph_builder.add_node("chatbot", chatbot)
# define entrypoint
graph_builder.add_edge(START, "chatbot")
# define exit point
graph_builder.add_edge("chatbot", END)

# Compile & run
graph = graph_builder.compile()
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
