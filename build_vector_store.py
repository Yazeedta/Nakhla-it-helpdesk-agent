
import os
import json
import hashlib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# Configuration
MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 20
VECTOR_DIMENSIONS = 1536

CHUNKS_PATH = Path("rag_store/chunks.json")
VECTORS_PATH = Path("rag_store/vectors.npz")


# Load the chunks prepared by ingestion.py
file_bytes = CHUNKS_PATH.read_bytes()
chunks = json.loads(file_bytes)

if not chunks:
    raise ValueError("No chunks found in chunks.json")

# Create a fingerprint to detect changes in the data
fingerprint = hashlib.sha256(
    MODEL.encode("utf-8") + file_bytes
).hexdigest()


# Reuse saved vectors when the data has not changed
if VECTORS_PATH.exists():
    with np.load(VECTORS_PATH, allow_pickle=False) as saved:
        cache_is_valid = (
            saved["model"].item() == MODEL
            and saved["fingerprint"].item() == fingerprint
            and saved["embeddings"].shape
            == (len(chunks), VECTOR_DIMENSIONS)
        )

    if cache_is_valid:
        print("Vector store is already up to date.")
        print("Total vectors:", len(chunks))
        raise SystemExit


# Prepare text for embedding
texts = []

for chunk in chunks:
    title = chunk["metadata"]["title"]

    text = (
        f"Document: {chunk['document_id']}\n"
        f"Title: {title}\n\n"
        f"{chunk['text']}"
    )

    texts.append(text)


# Connect to OpenRouter
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=90.0,
    max_retries=2
)


# Generate embeddings in batches
all_embeddings = []

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]

    response = client.embeddings.create(
        model=MODEL,
        input=batch
    )

    # Preserve the original order of the chunks
    ordered_vectors = [None] * len(batch)

    for item in response.data:
        ordered_vectors[item.index] = item.embedding

    if any(vector is None for vector in ordered_vectors):
        raise ValueError("Some embeddings are missing")

    all_embeddings.extend(ordered_vectors)

    print(
        f"Embedded {len(all_embeddings)}/{len(texts)} chunks"
    )


# Convert the vectors into a NumPy matrix
embeddings = np.asarray(all_embeddings, dtype=np.float32)

if embeddings.shape != (len(chunks), VECTOR_DIMENSIONS):
    raise ValueError(f"Unexpected vector shape: {embeddings.shape}")


# Save vectors and cache information
np.savez_compressed(
    VECTORS_PATH,
    embeddings=embeddings,
    model=MODEL,
    fingerprint=fingerprint
)

print("\nVector Store Created Successfully!")
print("Vector shape:", embeddings.shape)
print("Saved to:", VECTORS_PATH)

