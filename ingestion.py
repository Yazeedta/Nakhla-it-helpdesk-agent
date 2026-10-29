import re
import json
import pymupdf

from pathlib import Path
import yaml

# Folder containing the knowledge-base documents
kb_folder = Path("dataset/knowledge_base")

# Find all Markdown files
md_files = sorted(kb_folder.glob("*.md"))

documents = []

for file_path in md_files:
    # Read the document
    raw_text = file_path.read_text(encoding="utf-8")

    # Separate metadata from content
    parts = raw_text.split("---", 2)

    metadata = yaml.safe_load(parts[1])
    metadata["format"] = "md"
    content = parts[2].strip()

    # Keep metadata and text together
    document = {
        "metadata": metadata,
        "content": content
    }

    documents.append(document)

    print(metadata["id"], "-", metadata["status"])


# Read the document inventory from manifest.json
manifest_path = Path("dataset/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

# Select PDF documents from the manifest
pdf_documents = [
    item for item in manifest["documents"]
    if item["format"] == "pdf"
]

for metadata in pdf_documents:
    # Locate the PDF file
    pdf_path = Path("dataset") / metadata["file"]

    # Extract text from every PDF page
    with pymupdf.open(pdf_path) as pdf:
        pages = [page.get_text("text") for page in pdf]

    # Store the text together with its metadata
    document = {
        "metadata": metadata,
        "content": "\n\n".join(pages),
        "pages": pages
    }

    documents.append(document)

    print(metadata["id"], "-", metadata["status"],
          "-", len(pages), "pages")

print("Total documents:", len(documents))

# Validate that all documents contain extracted text
for doc in documents:
    doc_id = doc["metadata"]["id"]
    text_length = len(doc["content"])

    if text_length == 0:
        print("EMPTY DOCUMENT:", doc_id)

# Inspect the beginning of each PDF
for doc in documents:
   if doc["metadata"].get("format") == "pdf":
        print("\nPDF:", doc["metadata"]["id"])
        print("Text length:", len(doc["content"]))
        print("Preview:", doc["content"][:200])

        
# Split a Markdown document into sections
def chunk_by_section(document):
    content = document["content"]
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    chunks = []

    for section in sections:
        if not section.strip():
            continue

        chunk = {
            "document_id": document["metadata"]["id"],
            "metadata": document["metadata"].copy(),
            "text": section.strip()
        }

        chunks.append(chunk)

    return chunks


# Test chunking on KB-002
vpn_doc = next(
    doc for doc in documents
    if doc["metadata"]["id"] == "KB-002"
)

vpn_chunks = chunk_by_section(vpn_doc)

print("\nKB-002 chunks:", len(vpn_chunks))

for i, chunk in enumerate(vpn_chunks, start=1):
    print(f"Chunk {i}:", chunk["text"][:80])

print("\nFull Error Codes Section:")
print(vpn_chunks[4]["text"])


# Extract individual error-code rows from the table
error_section = vpn_chunks[4]["text"]

error_chunks = []

for line in error_section.splitlines():
    # Find table rows that start with an NX error code
    if not re.match(r"^\|\s*`NX-\d+`", line):
        continue

    # Split the row into code, meaning, and fix
    cells = line.strip().strip("|").split("|", 2)

    code = cells[0].strip().strip("`")
    meaning = cells[1].strip()
    fix = cells[2].strip()

    # Keep the complete fix and document metadata
    chunk = {
        "document_id": "KB-002",
        "metadata": vpn_doc["metadata"].copy(),
        "text": f"VPN Error {code}\nMeaning: {meaning}\nFix: {fix}"
    }

    error_chunks.append(chunk)

print("\nIndividual error chunks:", len(error_chunks))

for chunk in error_chunks:
    print("\n", chunk["text"])

# Replace the full error table with individual error chunks
vpn_chunks = vpn_chunks[:4] + error_chunks + vpn_chunks[5:]

print("Final KB-002 chunks:", len(vpn_chunks))

for chunk in vpn_chunks:
    print(chunk["text"].splitlines()[0])

    
# Build chunks for all knowledge-base documents
all_chunks = []

for doc in documents:
    doc_id = doc["metadata"]["id"]

    if doc_id == "KB-002":
        chunks = vpn_chunks

    elif doc["metadata"]["format"] == "pdf":
        chunks = []

        for page_number, page_text in enumerate(doc["pages"], start=1):
            if not page_text.strip():
                continue

            chunks.append({
                "document_id": doc_id,
                "metadata": doc["metadata"].copy(),
                "page": page_number,
                "text": page_text.strip()
            })

    else:
        chunks = chunk_by_section(doc)

    all_chunks.extend(chunks)

    print(f"{doc_id}: {len(chunks)} chunks")

print("\nTotal documents:", len(documents))
print("Total chunks:", len(all_chunks))


# Check chunk sizes before generating embeddings
chunk_sizes = [len(chunk["text"]) for chunk in all_chunks]

print("\nChunk Size Report")
print("Total chunks:", len(all_chunks))
print("Smallest chunk:", min(chunk_sizes))
print("Largest chunk:", max(chunk_sizes))
print("Average size:", round(sum(chunk_sizes) / len(chunk_sizes), 2))

# Inspect the five largest chunks
largest_chunks = sorted(
    all_chunks,
    key=lambda chunk: len(chunk["text"]),
    reverse=True
)[:5]

print("\nFive largest chunks:")

for chunk in largest_chunks:
    print(
        chunk["document_id"],
        "-",
        len(chunk["text"]),
        "characters",
        "-",
        chunk["text"].splitlines()[0][:70]
    )


# Prepare trusted chunks for retrieval
index_chunks = []

for i, chunk in enumerate(all_chunks, start=1):
    metadata = chunk["metadata"]

    # Exclude deprecated and draft documents
    if metadata.get("status") != "current":
        continue

    text = chunk["text"]

    # Add the document title to PDF chunks for better context
    if metadata.get("format") == "pdf":
        text = f"{metadata['title']}\n\n{text}"

    index_chunks.append({
        "chunk_id": f"chunk-{i:03d}",
        "document_id": chunk["document_id"],
        "text": text,
        "metadata": metadata,
        "page": chunk.get("page")
    })

# Save the chunks to a JSON file
output_folder = Path("rag_store")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "chunks.json"

output_path.write_text(
    json.dumps(
        index_chunks,
        ensure_ascii=False,
        indent=2,
        default=str
    ),
    encoding="utf-8"
)

print("\nRetrieval Index Preparation")
print("All chunks:", len(all_chunks))
print("Current chunks:", len(index_chunks))
print("Excluded chunks:", len(all_chunks) - len(index_chunks))
print("Saved to:", output_path)
