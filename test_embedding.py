
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

# Connect to OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Test the embedding model with one short text
response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input="VPN error NX-403: Access denied"
)

embedding = response.data[0].embedding

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First 5 values:", embedding[:5])

