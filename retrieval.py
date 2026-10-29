
import os
import re
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# Configuration
MODEL = "openai/text-embedding-3-small"

CHUNKS_PATH = Path("rag_store/chunks.json")
VECTORS_PATH = Path("rag_store/vectors.npz")


# Load saved chunks and embeddings
chunks = json.loads(
    CHUNKS_PATH.read_text(encoding="utf-8")
)

with np.load(VECTORS_PATH, allow_pickle=False) as data:
    embeddings = data["embeddings"]
    saved_model = data["model"].item()

if saved_model != MODEL:
    raise ValueError("Embedding model mismatch")

if len(chunks) != len(embeddings):
    raise ValueError("Chunks and vectors count mismatch")


# Connect to OpenRouter
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def search(query, top_k=5):
    if not query.strip():
        return []

    # Generate an embedding for the user's question
    response = client.embeddings.create(
        model=MODEL,
        input=query
    )

    query_vector = np.asarray(
        response.data[0].embedding,
        dtype=np.float32
    )

    # Calculate cosine similarity
    dot_products = embeddings @ query_vector

    chunk_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)

    similarities = dot_products / (
        chunk_norms * query_norm + 1e-10
    )

    # Extract exact error codes from the question
    error_codes = re.findall(
        r"\b[A-Z]{2,5}-\d{2,4}\b",
        query.upper()
    )

    results = []

    for i, chunk in enumerate(chunks):
        score = float(similarities[i])

        # Inspect the first line of the chunk
        first_line = chunk["text"].splitlines()[0].upper()

        # Boost exact error codes in chunk headings
        if error_codes:
            if any(
                re.search(rf"\b{re.escape(code)}\b", first_line)
                for code in error_codes
            ):
                score += 0.35

            elif "CHANGE LOG" in first_line:
                score -= 0.25

        results.append({
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "page": chunk.get("page"),
            "score": score
        })

    # Sort results from highest to lowest score
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# Test the retrieval system
if __name__ == "__main__":
    query = "How do I fix VPN error NX-403?"

    print("Question:", query)

    results = search(query, top_k=5)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Document:", result["document_id"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"][:500])
