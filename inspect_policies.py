
import json
from pathlib import Path

# Load the saved chunks without calling OpenRouter
chunks_path = Path("rag_store/chunks.json")
chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

# Find sections relevant to policy conflicts
checks = {
    "SP-002": ["sms", "voice", "backup code"],
    "KB-016": ["finance", "vpn"],
    "KB-002": ["finance", "exceptions"]
}

for document_id, keywords in checks.items():
    print(f"\n{'=' * 60}")
    print("Document:", document_id)

    for chunk in chunks:
        if chunk["document_id"] != document_id:
            continue

        text = chunk["text"]
        text_lower = text.lower()

        matched = [
            keyword for keyword in keywords
            if keyword in text_lower
        ]

        if matched:
            print("\nChunk:", chunk["chunk_id"])
            print("Matched keywords:", matched)
            print("Content:\n", text)

