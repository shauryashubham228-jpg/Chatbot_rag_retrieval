"""
Inspect what the retriever returns for a given query.

Usage:
    python inspect_retrieval.py "bank refund for item over Rs.5000"
    python inspect_retrieval.py            # runs a few sample queries
"""
from __future__ import annotations

import sys

from app.vector_store import build_retriever

SAMPLE_QUERIES = [
    "bank refund deduction for an item over Rs.5000",
    "what fee if I cancel my order after dispatch",
    "can I return innerwear",
    "how long does delivery take in India",
]


def show(retriever, query: str) -> None:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    docs = retriever.invoke(query)
    for i, d in enumerate(docs, 1):
        headers = " > ".join(str(v) for v in d.metadata.values()) or "(no header)"
        print(f"\n[{i}] {headers}")
        print(d.page_content.strip())


def main() -> None:
    retriever = build_retriever()
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_QUERIES
    for q in queries:
        show(retriever, q)


if __name__ == "__main__":
    main()
