import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PIPELINE_DIR = os.path.join(
    PROJECT_ROOT,
    "pipeline"
)

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from naive_rag import retrieve_documents


REFERENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "queries",
    "full_references.csv",
)


def main():

    df = pd.read_csv(REFERENCE_FILE)

    missing = df[
        df["reference_answer"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ]

    print("=" * 70)
    print("MISSING REFERENCE RETRIEVAL DIAGNOSTIC")
    print("=" * 70)

    print(f"Missing references: {len(missing)}")

    for _, row in missing.iterrows():

        query = str(row["query"]).strip()

        print("\n" + "-" * 70)
        print(f"Query: {query}")
        print("-" * 70)

        documents, metadatas, distances = retrieve_documents(
            query,
            top_k=10,
        )

        if not documents:
            print("No documents retrieved.")
            continue

        for i, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            print(f"\n[{i}] Source: {metadata.get('source', 'unknown')}")
            print(f"Distance: {distance:.4f}")
            print(document[:700].replace("\n", " "))


if __name__ == "__main__":
    main()