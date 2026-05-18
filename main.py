# main.py
# This is the entry point — just run this file to start chatting
#
# Before running:
#   1. pip install langchain langgraph langchain-anthropic langchain-pinecone langchain-openai
#   2. Create a .env file with your API keys (see below)
#   3. Run: python memories.py   ← adds sample memories first
#   4. Run: python main.py       ← start chatting

import os
from dotenv import load_dotenv
from graph import grandpa_graph

# ── Load API keys from .env file ───────────────────────────────────────────────
# Create a file called .env in the same folder with this content:
#
#   ANTHROPIC_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here       ← used for embeddings
#   PINECONE_API_KEY=your_key_here
#
load_dotenv()


# ── Main chat loop ─────────────────────────────────────────────────────────────

def chat():
    print("=" * 50)
    print("  💬 Memorial AI — Talk to Grandpa Wei")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    while True:
        # Get input from user
        user_input = input("You: ").strip()

        # Exit if user types quit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye. 🙏")
            break

        # Skip empty input
        if not user_input:
            continue

        # Run the graph with the user's message
        # The graph will: find memories → generate reply
        result = grandpa_graph.invoke({
            "user_message": user_input,
            "relevant_memories": "",  # starts empty, filled by rag_node
            "reply": ""               # starts empty, filled by llm_node
        })

        # Print Grandpa's reply
        print(f"\nGrandpa: {result['reply']}\n")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chat()
