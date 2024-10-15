from functools import partial
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

_RED = "\033[91m"
_ORANGE = "\033[93m"
_ENDC = "\033[0m"


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o", max_tokens=150, temperature=0.8)


def agent(state: State, sys_prompt: dict[str, str]):
    return {"messages": [llm.invoke([("system", sys_prompt)] + state["messages"])]}


prompt1 = (
    "You are a crazy scientist that can't stand flat earthers."
    "You are trying to explain to them why the earth is round while being very angry."
    "Answer in at most 2 sentences, reflecting on the opposition's arguments."
    "Use reasoning and scientific facts to support your argument."
    "Come up with new arguments each time"
)

prompt2 = (
    "You are a very calm flat earther that is trying to convince a people that the earth is flat."
    "Answer in at most 2 sentences, reflecting on the opposition's arguments."
    "Use reasoning and common flat-earth arguments to support your argument."
)

agent1 = partial(agent, sys_prompt=prompt1)
agent2 = partial(agent, sys_prompt=prompt2)

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)

graph_builder.add_edge(START, "agent1")
graph_builder.add_edge("agent1", "agent2")
MAX_TURNS = 10
graph_builder.add_conditional_edges(
    "agent2",
    lambda state: "agent1" if len(state["messages"]) < MAX_TURNS else END,
)


# Compile & run
TURN = 0
graph = graph_builder.compile()
for event in graph.stream({"messages": [("ai", "I think the earth is flat.")]}):
    for value in event.values():
        if TURN % 2 == 0:
            print(f"{_RED}Scientist:{_ENDC}", value["messages"][-1].content)
        else:
            print(f"{_ORANGE}Flat-earther:{_ENDC}", value["messages"][-1].content)
        TURN += 1
