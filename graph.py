# graph.py
# This file builds the conversation flow using LangGraph
# Flow: User message → Find memories → Groq replies

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from typing import TypedDict

from memories import search_memory  # our memories.py file

# ── State ──────────────────────────────────────────────────────────────────────

# State is like a shared notepad passed between each step
# Every node can read from it and write to it
class GrandpaState(TypedDict):
    user_message: str       # what the user typed
    relevant_memories: str  # memories found by RAG
    reply: str              # Grandpa's final reply

# ── Groq setup ─────────────────────────────────────────────────────────────────

# Groq is very fast and free to start — get your key at console.groq.com
# llama-3.3-70b-versatile is a great free model for conversational AI
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── The 2 Nodes ────────────────────────────────────────────────────────────────

def rag_node(state: GrandpaState) -> GrandpaState:
    """
    Node 1: Find relevant memories.

    Takes the user's message and searches Pinecone for
    the most related memories about Grandpa.

    Example:
        User asks: "Did you enjoy teaching?"
        → Finds: "Worked as a schoolteacher for 35 years..."
    """
    print("🔍 Searching memories...")

    memories = search_memory(state["user_message"])

    # Save the found memories into state for the next node
    return {"relevant_memories": memories}


def llm_node(state: GrandpaState) -> GrandpaState:
    """
    Node 2: Generate Grandpa's reply using Groq.

    Gives Groq the relevant memories so it can answer
    as if it IS Grandpa — not just a generic AI.
    """
    print("🧠 Grandpa is thinking...")

    # This prompt tells the model how to behave
    system_prompt = f"""You are Wei, a wise and warm elderly grandfather.
Speak in first person, as if you ARE this person — not an AI.
Keep your replies short and natural, like real conversation (2-4 sentences).
Stay warm, gentle, and wise.

Here are some things you remember about your life:
{state['relevant_memories']}

Never say you are an AI. Never break character."""

    # Send the conversation to Groq
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_message"])
    ]

    response = llm.invoke(messages)

    # Save the reply into state
    return {"reply": response.content}


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_graph():
    """
    Connect the nodes together in order:
        START → rag_node → llm_node → END
    """
    graph = StateGraph(GrandpaState)

    # Add each node (give it a name and the function to run)
    graph.add_node("find_memories", rag_node)
    graph.add_node("generate_reply", llm_node)

    # Connect them in order
    graph.set_entry_point("find_memories")           # start here
    graph.add_edge("find_memories", "generate_reply") # then go here
    graph.add_edge("generate_reply", END)             # then finish

    return graph.compile()


# Create the graph (ready to use)
grandpa_graph = build_graph()
