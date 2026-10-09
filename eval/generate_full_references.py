import os
import sys
import time

import pandas as pd
import chromadb
from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PIPELINE_DIR = os.path.join(
    PROJECT_ROOT,
    "pipeline",
)

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from naive_rag import retrieve_documents


load_dotenv()


CHROMA_PATH = os.path.join(
    PROJECT_ROOT,
    "chroma_db",
)

COLLECTION_NAME = "wikipedia_rag"

QUERY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "queries",
    "evaluation_queries.csv",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "queries",
    "full_references.csv",
)

MODEL = "openai/gpt-oss-20b"


def clean_cell(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def load_resources():

    print("\nLoading ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection loaded: {COLLECTION_NAME}"
    )

    print(
        "\nConnecting to Groq..."
    )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env"
        )

    client = Groq(
        api_key=api_key
    )

    print(
        "Groq connection ready."
    )

    return collection, client


def generate_reference(
    query,
    client,
):

    documents, metadatas, distances = retrieve_documents(
    query,
    top_k=10,
)

    if not documents:
        return ""

    context = "\n\n".join(
        [
            f"[{i + 1}] {document}"
            for i, document in enumerate(documents)
        ]
    )

    prompt = f"""
Create a concise ground-truth reference answer
for the question below using ONLY the provided
Wikipedia corpus context.

The answer will be manually reviewed later.

Requirements:

- Answer the question directly.
- Include the important facts needed to answer it.
- For comparison questions, clearly describe both sides
  and their key differences.
- For multi-hop questions, connect the relevant concepts.
- Do not invent information that is not supported by
  the context.
- Do not mention that you are using context.
- Do not use citations or source numbers.
- Keep the answer concise but complete.
- If the context is insufficient to answer the question,
  return an empty answer instead of explaining that the
  context is insufficient.

Context:
{context}

Question:
{query}

Reference answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
    )

    return (
        response.choices[0]
        .message
        .content
        .strip()
    )


def save_results(results):

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main():

    print("=" * 70)

    print(
        "CP4.7.1 - RESUME FULL REFERENCE DRAFT GENERATION"
    )

    print("=" * 70)

    # ============================================================
    # LOAD OFFICIAL QUERY DATASET
    # ============================================================

    queries_df = pd.read_csv(
        QUERY_FILE
    )

    required_columns = {
        "query",
        "expected_difficulty_label",
    }

    missing = (
        required_columns
        - set(queries_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(queries_df) != 200:
        raise ValueError(
            f"Expected 200 queries, "
            f"found {len(queries_df)}"
        )

    print(
        f"\nTotal queries: "
        f"{len(queries_df)}"
    )

    print(
        "\nDistribution:"
    )

    print(
        queries_df[
            "expected_difficulty_label"
        ].value_counts()
    )

    # ============================================================
    # LOAD EXISTING REFERENCES
    # ============================================================

    existing_map = {}

    if os.path.exists(OUTPUT_FILE):

        print(
            "\nExisting reference file found."
        )

        existing_df = pd.read_csv(
            OUTPUT_FILE
        )

        print(
            f"Existing rows: "
            f"{len(existing_df)}"
        )

        required_output_columns = {
            "query",
            "reference_answer",
            "review_status",
        }

        missing_output = (
            required_output_columns
            - set(existing_df.columns)
        )

        if missing_output:
            raise ValueError(
                f"Existing reference file is missing "
                f"columns: {missing_output}"
            )

        # --------------------------------------------------------
        # MATCH EXISTING ANSWERS BY QUERY TEXT
        # --------------------------------------------------------

        for _, row in existing_df.iterrows():

            existing_query = clean_cell(
                row["query"]
            )

            existing_answer = clean_cell(
                row["reference_answer"]
            )

            if (
                existing_query
                and existing_answer
            ):

                existing_map[
                    existing_query
                ] = existing_answer

        print(
            f"Existing non-empty references: "
            f"{len(existing_map)}"
        )

    else:

        print(
            "\nNo existing reference file found."
        )

        print(
            "Starting from Query 1."
        )

    # ============================================================
    # REBUILD CANONICAL 200-ROW DATASET
    # ============================================================

    results = []

    for _, row in queries_df.iterrows():

        query = clean_cell(
            row["query"]
        )

        if query in existing_map:

            results.append(
                {
                    "query": query,
                    "reference_answer":
                        existing_map[query],
                    "review_status":
                        "drafted",
                }
            )

        else:

            results.append(
                {
                    "query": query,
                    "reference_answer": "",
                    "review_status":
                        "needs_review",
                }
            )

    # ============================================================
    # CHECK CURRENT PROGRESS
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    completed_mask = (
        results_df[
            "reference_answer"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    completed_count = int(
        completed_mask.sum()
    )

    remaining_count = (
        len(results_df)
        - completed_count
    )

    print(
        "\nCurrent progress:"
    )

    print(
        f"Completed references: "
        f"{completed_count}"
    )

    print(
        f"Remaining references: "
        f"{remaining_count}"
    )

    # ------------------------------------------------------------
    # SAVE CLEAN CANONICAL FILE
    # ------------------------------------------------------------

    save_results(
        results
    )

    # ============================================================
    # NOTHING LEFT TO GENERATE
    # ============================================================

    if remaining_count == 0:

        print(
            "\nAll 200 references already exist."
        )

        print(
            "No Groq calls are required."
        )

        return

    # ============================================================
    # LOAD RESOURCES ONLY WHEN GENERATION IS REQUIRED
    # ============================================================

    _, client = load_resources()

    # ============================================================
    # GENERATE MISSING REFERENCES
    # ============================================================

    print(
        "\nStarting/resuming draft generation..."
    )

    for index, row in queries_df.iterrows():

        query = clean_cell(
            row["query"]
        )

        difficulty = clean_cell(
            row[
                "expected_difficulty_label"
            ]
        )

        current_reference = clean_cell(
            results[index][
                "reference_answer"
            ]
        )

        # --------------------------------------------------------
        # SKIP EXISTING REFERENCE
        # --------------------------------------------------------

        if current_reference:

            print(
                "\n" + "-" * 70
            )

            print(
                f"Query {index + 1}/"
                f"{len(queries_df)}"
            )

            print(
                "Status: already drafted - SKIPPING"
            )

            continue

        # --------------------------------------------------------
        # GENERATE MISSING REFERENCE
        # --------------------------------------------------------

        print(
            "\n" + "-" * 70
        )

        print(
            f"Query {index + 1}/"
            f"{len(queries_df)}"
        )

        print(
            f"Category: {difficulty}"
        )

        print(
            f"Question: {query}"
        )

        try:

            start_time = time.perf_counter()

            reference_answer = generate_reference(
                query,
                client,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            reference_answer = clean_cell(
                reference_answer
            )

            if reference_answer:

                status = "drafted"

                print(
                    f"Generated successfully "
                    f"({elapsed:.2f}s)"
                )

            else:

                status = "needs_review"

                print(
                    "No valid reference generated."
                )

        except Exception as error:

            error_message = str(
                error
            )

            print(
                f"ERROR: {error_message}"
            )

            # ----------------------------------------------------
            # RATE LIMIT DETECTION
            # ----------------------------------------------------

            if (
                "429" in error_message
                or "rate_limit_exceeded"
                in error_message
                or "Rate limit reached"
                in error_message
            ):

                print(
                    "\n" + "=" * 70
                )

                print(
                    "GROQ RATE LIMIT DETECTED"
                )

                print(
                    "=" * 70
                )

                print(
                    "\nStopping safely."
                )

                print(
                    "All completed references "
                    "have been preserved."
                )

                save_results(
                    results
                )

                current_completed = sum(
                    1
                    for item in results
                    if clean_cell(
                        item["reference_answer"]
                    )
                )

                print(
                    f"\nCompleted references: "
                    f"{current_completed}"
                )

                print(
                    f"Remaining references: "
                    f"{len(results) - current_completed}"
                )

                print(
                    "\nRun the same script again "
                    "when the Groq quota is available."
                )

                print(
                    "Completed queries will be "
                    "automatically skipped."
                )

                return

            # ----------------------------------------------------
            # NON-RATE-LIMIT ERROR
            # ----------------------------------------------------

            reference_answer = ""

            status = "needs_review"

        # --------------------------------------------------------
        # UPDATE RESULT
        # --------------------------------------------------------

        results[index] = {
            "query": query,
            "reference_answer":
                reference_answer,
            "review_status":
                status,
        }

        # --------------------------------------------------------
        # SAVE AFTER EVERY QUERY
        # --------------------------------------------------------

        save_results(
            results
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n\n" + "=" * 70
    )

    print(
        "CP4.7.1 SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal queries: "
        f"{len(results_df)}"
    )

    print(
        "\nReview status:"
    )

    print(
        results_df[
            "review_status"
        ].value_counts()
    )

    empty_references = (
        results_df[
            "reference_answer"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "\nEmpty references:"
    )

    print(
        empty_references
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nCP4.7.1 draft generation complete."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are DRAFT references only."
    )

    print(
        "They must be manually reviewed before CP5."
    )


if __name__ == "__main__":
    main()