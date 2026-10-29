
from retrieval import search

# Test retrieval across different IT Helpdesk scenarios
test_cases = [
    {
        "query": "I work in Finance and my VPN shows NX-403. What should I do?",
        "expected_docs": ["KB-002", "KB-016"]
    },
    {
        "query": "Can the IT Service Desk reset my password over the phone?",
        "expected_docs": ["SP-001", "KB-004"]
    },
    {
        "query": "I lost my phone. Can I receive an MFA code by SMS?",
        "expected_docs": ["SP-002", "KB-007"]
    },
    {
        "query": "How do I troubleshoot NakhlaPrint error PRN-21?",
        "expected_docs": ["KB-013"]
    }
]


for case in test_cases:
    query = case["query"]

    print("\n" + "=" * 60)
    print("QUESTION:", query)

    results = search(query, top_k=10)

    print("\nTop 5 results:")

    for rank, result in enumerate(results[:5], start=1):
        heading = result["text"].splitlines()[0][:70]

        print(
            f"{rank}. {result['document_id']} "
            f"| Score: {result['score']:.4f} "
            f"| {heading}"
        )

    print("\nExpected document rankings:")

    for doc_id in case["expected_docs"]:
        rank = next(
            (
                i
                for i, result in enumerate(results, start=1)
                if result["document_id"] == doc_id
            ),
            None
        )

        if rank is None:
            print(f"{doc_id}: Not found in top 10")
        else:
            print(f"{doc_id}: Rank {rank}")

