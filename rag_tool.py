import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from crewai.tools import tool

load_dotenv()
client = OpenAI()

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "faq.txt")


def load_chunks():
    with open(KB_PATH, encoding="utf-8") as f:
        text = f.read()
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


# Build the "vector store" once, when the file is imported
CHUNKS = load_chunks()
CHUNK_VECTORS = embed(CHUNKS)


@tool("Company Knowledge Base Search")
def search_knowledge_base(query: str) -> str:
    """Search ShopEase's FAQ and policy documents. Returns the most relevant passages for a customer question."""
    q = embed([query])[0]
    scores = CHUNK_VECTORS @ q / (np.linalg.norm(CHUNK_VECTORS, axis=1) * np.linalg.norm(q))
    top = scores.argsort()[::-1][:3]
    return "\n\n---\n\n".join(CHUNKS[i] for i in top)