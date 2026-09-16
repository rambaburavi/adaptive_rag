import csv
import sys
from pathlib import Path

# Allow importing from the pipeline directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.naive_rag import retrieve_documents, generate_answer, build_prompt


QUERY_FILE = PROJECT_ROOT / "data" / "queries" / "evaluation_queries.csv"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "sanity_results.csv"

SIMPLE_COUNT = 5
MULTIHOP_COUNT = 5
COMPARISON_COUNT = 5


def load_queries():
    """Load evaluation queries from CSV."""

    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = []

    for label, count in [
        ("simple", SIMPLE_COUNT),
        ("multi-hop", MULTIHOP_COUNT),
        ("comparison", COMPARISON_COUNT),
    ]:
        matching = [
            row for row in rows
            if row["expected_difficulty_label"] == label
        ]

        selected.extend(matching[:count])

    return selected


def run_test(query_row):
    """Run one query through the naive RAG pipeline."""

    query = query_row["query"]
    difficulty = query_row["expected_difficulty_label"]
    expected_sources = query_row["expected_sources"]

    print("\n" + "=" * 70)
    print(f"Difficulty: {difficulty}")
    print(f"Query: {query}")
    print("=" * 70)

    # Retrieve chunks
    documents, metadatas, distances = retrieve_documents(query)

    print("\nRetrieved sources:")

    sources = []

    for i, metadata in enumerate(metadatas, start=1):
        source = metadata.get("source", "Unknown")
        distance = distances[i - 1]

        sources.append(source)

        print(
            f"{i}. {source} "
            f"(distance: {distance:.4f})"
        )

    # Build prompt
    prompt = build_prompt(
        query,
        documents,
        metadatas
    )

    # Generate answer
    answer = generate_answer(prompt)

    print("\nAnswer:")
    print(answer)

    return {
        "query": query,
        "difficulty": difficulty,
        "expected_sources": expected_sources,
        "retrieved_sources": " | ".join(sources),
        "answer": answer.replace("\n", " "),
    }


def main():

    print("=" * 70)
    print("NAIVE RAG — CHECKPOINT 1 SANITY TEST")
    print("=" * 70)

    queries = load_queries()

    print(f"\nSelected queries: {len(queries)}")

    results = []

    for query in queries:

        try:
            result = run_test(query)
            results.append(result)

        except Exception as e:

            print(f"\nERROR processing query: {query['query']}")
            print(e)

    # Save results
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "query",
            "difficulty",
            "expected_sources",
            "retrieved_sources",
            "answer",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("SANITY TEST COMPLETE")
    print("=" * 70)

    print(f"Queries tested : {len(results)}")
    print(f"Results saved  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()