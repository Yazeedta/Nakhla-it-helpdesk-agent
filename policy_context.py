
from retrieval import chunks, search


def find_policy_chunk(document_id, heading):
    """Find a current document chunk by its heading."""

    for chunk in chunks:
        if (
            chunk["document_id"] == document_id
            and chunk["metadata"].get("status") == "current"
            and chunk["text"].lstrip().startswith(heading)
        ):
            return chunk

    raise ValueError(
        f"Required policy section not found: {document_id} / {heading}"
    )


def get_policy_context(query, department=None, top_k=5):
    """Combine mandatory policy evidence with semantic retrieval."""

    query_lower = query.lower()
    department_lower = (department or "").lower()

    required = []

    # MFA rules: SMS and voice codes are forbidden
    if any(
        word in query_lower
        for word in ("mfa", "sms", "voice code", "backup code")
    ):
        required.append(
            ("SP-002", "## 1. Multi-factor authentication")
        )

    # Distinguish lost/stolen devices from phones left at home
    if (
        any(word in query_lower for word in ("lost", "stolen", "forgot"))
        and any(word in query_lower for word in ("phone", "device", "laptop"))
    ):
        required.append(
            ("SP-002", "## 2. Lost or stolen devices")
        )

    # Finance VPN exception
    if (
        department_lower == "finance"
        and any(
            word in query_lower
            for word in ("vpn", "remote access", "nx-")
        )
    ):
        required.extend([
            ("KB-016", "## Matrix"),
            ("KB-002", "## Exceptions")
        ])

    # Password reset security policy
    if (
        "password" in query_lower
        and any(
            word in query_lower
            for word in ("phone", "call", "reset")
        )
    ):
        required.append(
            ("SP-001", "## 2. Password resets")
        )

    evidence = []
    seen_ids = set()

    # Include relevant mandatory policy sections first
    for document_id, heading in required:
        chunk = find_policy_chunk(document_id, heading)

        if chunk["chunk_id"] not in seen_ids:
            evidence.append({
                **chunk,
                "reason": "required_policy"
            })
            seen_ids.add(chunk["chunk_id"])

    # Add semantic retrieval results without duplication
    results = search(query, top_k=top_k)

    for result in results:
        if result["chunk_id"] not in seen_ids:
            evidence.append({
                **result,
                "reason": "semantic_retrieval"
            })
            seen_ids.add(result["chunk_id"])

    return evidence


if __name__ == "__main__":
    tests = [
        {
            "query": "I work in Finance and my VPN shows NX-403.",
            "department": "Finance"
        },
        {
            "query": "I lost my phone. Can I receive an MFA code by SMS?",
            "department": None
        }
    ]

    for test in tests:
        print("\n" + "=" * 60)
        print("Question:", test["query"])

        evidence = get_policy_context(
            test["query"],
            department=test["department"]
        )

        for item in evidence:
            print(
                f"{item['document_id']} | "
                f"{item['reason']} | "
                f"{item['text'].splitlines()[0]}"
            )

