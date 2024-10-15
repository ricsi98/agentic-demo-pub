from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([add])


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools=[add])


# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue)
graph_builder.add_edge("tools", "chatbot")

# Proper way of keeping state between calls | allow parallel calls
memory = MemorySaver()

# Compile & run
graph = graph_builder.compile(checkpointer=memory)
final_state = graph.invoke(
    {"messages": [HumanMessage(content=input("User: "))]},
    config={"configurable": {"thread_id": 42}},  # memory is bound to the thread_id
)
for message in final_state["messages"]:
    message.pretty_print()
