# memories.py
# This file handles storing and finding memories about Grandpa
# Think of it like a smart notebook that can search itself

import os

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Setup ──────────────────────────────────────────────────────────────────────

# Embeddings turn text into numbers so Pinecone can compare them
# Example: "loves fishing" becomes [0.23, -0.87, 0.54, ...]
# Using HuggingFace — completely free, no API key needed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to your Pinecone index (create one at pinecone.io, free tier works)
# Index name: "grandpa-memories", Dimensions: 384  ← note: 384 not 1536 for this model
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found. Make sure .env exists and contains PINECONE_API_KEY=...")

INDEX_NAME = "grandpa-memories"

pc = Pinecone(api_key=pinecone_api_key)
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key,
)

# ── Functions ──────────────────────────────────────────────────────────────────

def add_memory(text: str):
    """
    Save a new memory about Grandpa.

    Example:
        add_memory("Grandpa loved fishing at dawn every Sunday")
        add_memory("His wife was named Mei. She passed in 2015.")
        add_memory("He always said: hard work is the only honest road")
    """
    doc = Document(page_content=text)
    vector_store.add_documents([doc])
    print(f"✅ Memory saved: {text}")


def search_memory(query: str, top_k: int = 3) -> str:
    """
    Find the most relevant memories for a given question.

    Example:
        search_memory("Does Grandpa like fishing?")
        → Returns: "Grandpa loved fishing at dawn every Sunday"

    top_k = how many memories to return (3 is enough for most questions)
    """
    results = vector_store.similarity_search(query, k=top_k)

    # Join the top memories into one block of text
    memories = "\n".join([r.page_content for r in results])
    return memories


# ── Quick test (run this file directly to add sample memories) ─────────────────
if __name__ == "__main__":
    print("Adding sample memories about Grandpa Wei...")

    sample_memories = [
        "Born in 1938 in Shandong province. Loved fishing at dawn.",
        "Worked as a schoolteacher for 35 years. Very proud of his students.",
        "Wife's name was Mei. She passed away in 2015. He still talks about her.",
        "Favorite food: pork and cabbage dumplings, made every Sunday.",
        "Always said: 'Hard work is the only honest road.'",
        "Loved traditional Peking opera, especially the erhu instrument.",
        "Had 3 children: Liang, Fang, and Xiao. 7 grandchildren total.",
        "Collected old coins. Had a special brass coin from 1949.",
        "Passed away peacefully in March 2023 at age 84.",
    ]

    for memory in sample_memories:
        add_memory(memory)

    print("\n🔍 Test search: 'What did Grandpa like to do?'")
    print(search_memory("What did Grandpa like to do?"))
