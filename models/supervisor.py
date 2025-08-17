from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# Define the structure of the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Router function to determine the next node
def router(state: AgentState) -> str:
    user_query = state["messages"][-1].content.lower()

    # Check if the question is related to osteosarcoma
    if any(keyword in user_query for keyword in ["osteosarcoma", "bone cancer", "bone tumor", "sarcoma", "tumor"]):
        print(" Routing to RAG node")
        return "RAG"
    else:
        print("Routing to LLM node")
        return "LLM"