from langgraph.graph import END, START, StateGraph
from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain_ollama import ChatOllama
from langgraph.graph.message import MessagesState
from types import FunctionType
from typing import Annotated, List, Dict
from langgraph.types import Command

# ---- Tools ----
@tool
def addition(a: int, b: int) -> int:
    '''Adds two numbers together'''
    return a + b

@tool
def subtraction(a: int, b: int) -> int:
    '''Subtracts two numbers'''
    return a - b

@tool
def multiplication(a: int, b: int) -> int:
    '''Multiplies two numbers'''
    return a * b

@tool
def division(a: int, b: int) -> int:
    '''Divides two numbers'''
    if b == 0:
        return 0
    return a // b

data = {"a": 10, "b": 5, "c": 2, "d": 34}

@tool
def get_data(key1: str, key2: str) -> List[int]:
    '''Fetches values for two keys from the internal data store.'''
    return [data[key1], data[key2]]

def create_handoff_tool(agent_name: str, description=None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """{description}"""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    handoff_tool.__name__ = name
    handoff_tool.__doc__ = description
    return tool(handoff_tool)


# ---- LLM ----
ollama_model_tag = "qwen3:latest"
llm = ChatOllama(model=ollama_model_tag, base_url='http://127.0.0.1:11434')

# ---- Agents ----
math_agent = create_react_agent(
    model=llm,
    tools=[addition, subtraction, multiplication, division],
    name="math_agent",
    prompt="You are a math agent. Only assist with math problems. Reply with results only."
)

data_agent = create_react_agent(
    model=llm,
    tools=[get_data],
    name="data_agent",
    prompt="You are a data agent. Only assist with data lookup problems. Reply with results only."
)

assign_to_data_agent = create_handoff_tool("data_agent", "Assign task to the data agent.")
assign_to_math_agent = create_handoff_tool("math_agent", "Assign task to the math agent.")

supervisor_agent = create_react_agent(
    model=llm,
    tools=[assign_to_data_agent, assign_to_math_agent],
    name="supervisor",
    prompt=(
        "You are a supervisor. Delegate tasks to:\n"
        "- data agent (for information lookup)\n"
        "- math agent (for math problems)\n"
        "Delegate one task at a time. Do not perform work yourself."
    )
)

# ---- Graph ----
graph = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor_agent, destinations=("data_agent", "math_agent", END))
    .add_node("data_agent", data_agent)
    .add_node("math_agent", math_agent)
    .add_edge(START, "supervisor")
    .add_edge("data_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)

# ---- Display helpers ----
def pretty_print_message(message, indent=False):
    pretty = message.pretty_repr(html=True)
    if indent:
        pretty = "\n".join("\t" + line for line in pretty.splitlines())
    print(pretty)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        print(f"Update from subgraph {ns[-1].split(':')[0]}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        label = f"Update from node {node_name}:"
        if is_subgraph:
            label = "\t" + label
        print(label + "\n")
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]
        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()

# ---- Interaction Loop ----
if __name__ == "__main__":
    while True:
        user_input = input("Enter a message (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        for chunk in graph.stream(
            {
                "messages": [{"role": "user", "content": user_input}]
            }
        ):
            pretty_print_messages(chunk, last_message=True)
