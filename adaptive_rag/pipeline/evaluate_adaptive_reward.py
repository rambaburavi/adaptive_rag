import os
import sys
import time

import pandas as pd
import chromadb
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
sys.path.append(PIPELINE_DIR)


from actions import ACTIONS
from actions.common import new_state
from adaptive_router import adaptive_route
from reward.reward_function import compute_reward


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
REFERENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "queries",
    "checkpoint3_references.csv",
)
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint3_reward_results.csv",
)

QUERIES_PER_CATEGORY = 5



def main():

    print("=" * 70)
    print("CHECKPOINT 3 - REAL ADAPTIVE REWARD EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Groq
    # ---------------------------------------------------------

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Check your .env file."
        )

    # ---------------------------------------------------------
    # ChromaDB
    # ---------------------------------------------------------

    print("\nLoading ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection loaded: "
        f"{COLLECTION_NAME}"
    )

    # ---------------------------------------------------------
    # Groq client
    # ---------------------------------------------------------

    print("\nConnecting to Groq...")

    groq_client = Groq(
        api_key=groq_api_key
    )

    print("Groq connection ready.")

    # ---------------------------------------------------------
    # Evaluation queries
    # ---------------------------------------------------------

    print("\nLoading evaluation queries...")

    df = pd.read_csv(QUERY_FILE)
    
    reference_df = pd.read_csv(REFERENCE_FILE)

    required_reference_columns = {
        "query",
        "reference_answer",
    }

    missing_columns = (
        required_reference_columns
        - set(reference_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in reference file: "
            f"{missing_columns}"
        )

    reference_map = dict(
        zip(
            reference_df["query"].astype(str).str.strip(),
            reference_df["reference_answer"].astype(str),
        )
    )

    print(
        f"Ground-truth references loaded: "
        f"{len(reference_map)}"
    )

    evaluation_df = (
        df.groupby(
            "expected_difficulty_label",
            sort=False,
        )
        .head(QUERIES_PER_CATEGORY)
        .reset_index(drop=True)
    )

    print(
        f"Queries selected: "
        f"{len(evaluation_df)}"
    )

    print("\nDistribution:")

    print(
        evaluation_df[
            "expected_difficulty_label"
        ].value_counts()
    )

    results = []

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------

    for index, row in evaluation_df.iterrows():

        query = str(row["query"])

        expected_label = str(
            row["expected_difficulty_label"]
        )

        expected_sources = str(
            row.get("expected_sources", "")
        )

        print("\n" + "=" * 70)
        print(
            f"QUERY {index + 1}/"
            f"{len(evaluation_df)}"
        )
        print("=" * 70)

        print(f"Query: {query}")

        print(
            f"Expected category: "
            f"{expected_label}"
        )

        print(
            f"Expected sources: "
            f"{expected_sources}"
        )

        # -----------------------------------------------------
        # Adaptive routing
        # -----------------------------------------------------

        action_name, features = adaptive_route(
            query
        )

        print(
            f"Chosen action: "
            f"{action_name}"
        )

        print(
            f"Complexity score: "
            f"{features['complexity_score']}"
        )

        if action_name not in ACTIONS:

            print(
                f"ERROR: Action "
                f"'{action_name}' not found."
            )

            results.append({
                "query": query,
                "expected_label": expected_label,
                "expected_sources": expected_sources,
                "chosen_action": action_name,
                "complexity_score": features[
                    "complexity_score"
                ],
                "answer": "",
                "num_evidence": 0,
                "answer_quality": 0.0,
                "faithfulness": 0.0,
                "retrieval_precision": None,
                "latency_seconds": 0.0,
                "latency_score": 0.0,
                "reward": 0.0,
                "status": "action_not_found",
            })

            continue
        # -----------------------------------------------------
        # Ground-truth reference
        # -----------------------------------------------------

        reference_text = reference_map.get(
            query.strip(),
            "",
        )

        if not reference_text:
            print(
                "WARNING: No ground-truth reference found "
                "for this query."
            )
        else:
            print(
                "Ground-truth reference loaded."
            )

        # -----------------------------------------------------
        # Execute selected action
        # -----------------------------------------------------

        action_fn = ACTIONS[action_name]

        state = new_state()

        print("Executing action...")

        start_time = time.perf_counter()

        try:

            state, answer = action_fn(
                query,
                collection,
                groq_client,
                state,
            )

            status = "success"

        except Exception as error:

            end_time = time.perf_counter()

            latency_seconds = (
                end_time - start_time
            )

            print(
                f"ERROR while executing action: "
                f"{error}"
            )

            results.append({
                "query": query,
                "expected_label": expected_label,
                "expected_sources": expected_sources,
                "chosen_action": action_name,
                "complexity_score": features[
                    "complexity_score"
                ],
                "answer": "",
                "num_evidence": 0,
                "answer_quality": 0.0,
                "faithfulness": 0.0,
                "retrieval_precision": None,
                "latency_seconds": round(
                    latency_seconds,
                    4,
                ),
                "latency_score": 0.0,
                "reward": 0.0,
                "status": "action_error",
            })

            continue

        end_time = time.perf_counter()

        latency_seconds = (
            end_time - start_time
        )

        evidence = state.get(
            "evidence",
            [],
        )

        # -----------------------------------------------------
        # Compute reward
        # -----------------------------------------------------

        reward_result = compute_reward(
            query=query,
            answer=answer,
            evidence_chunks=evidence,
            latency_seconds=latency_seconds,
            action_name=action_name,
            reference_text=reference_text,
        )

        # -----------------------------------------------------
        # Print result
        # -----------------------------------------------------

        print(
            f"Latency: "
            f"{latency_seconds:.2f} seconds"
        )

        print(
            f"Evidence chunks: "
            f"{len(evidence)}"
        )

        print(
            f"Answer quality: "
            f"{reward_result['answer_quality']:.4f}"
        )

        if reward_result[
            "retrieval_precision"
        ] is None:

            print(
                "Retrieval precision: N/A"
            )

        else:

            print(
                f"Retrieval precision: "
                f"{reward_result['retrieval_precision']:.4f}"
            )

        if reward_result[
            "faithfulness"
        ] is None:

            print(
                "Faithfulness: N/A"
            )

        else:

            print(
                f"Faithfulness: "
                f"{reward_result['faithfulness']:.4f}"
            )

        print(
            f"Latency score: "
            f"{reward_result['latency_score']:.4f}"
        )

        print(
            f"REWARD: "
            f"{reward_result['reward']:.4f}"
        )

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        results.append({

            "query": query,

            "expected_label": expected_label,

            "expected_sources": expected_sources,

            "chosen_action": action_name,

            "complexity_score": features[
                "complexity_score"
            ],

            "answer": answer,

            "num_evidence": len(evidence),

            "answer_quality": reward_result[
                "answer_quality"
            ],

            "faithfulness": reward_result[
                "faithfulness"
            ],

            "retrieval_precision": reward_result[
                "retrieval_precision"
            ],

            "latency_seconds": reward_result[
                "latency_seconds"
            ],

            "latency_score": reward_result[
                "latency_score"
            ],

            "reward": reward_result[
                "reward"
            ],

            "status": status,
        })

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n\n")

    print("=" * 70)
    print("CHECKPOINT 3.5 - GROUND-TRUTH REWARD EVALUATION")
    print("=" * 70)

    print(
        f"\nTotal queries: "
        f"{len(results_df)}"
    )

    if not results_df.empty:

        successful_results = (
            results_df[
                results_df["status"]
                == "success"
            ]
        )

        print(
            f"Successful queries: "
            f"{len(successful_results)}"
        )

        if not successful_results.empty:

            print(
                f"\nAverage answer quality: "
                f"{successful_results['answer_quality'].mean():.4f}"
            )

            print(
                f"Average faithfulness: "
                f"{successful_results['faithfulness'].dropna().mean():.4f}"
            )

            print(
                f"Average retrieval precision: "
                f"{successful_results['retrieval_precision'].dropna().mean():.4f}"
            )

            print(
                f"Average latency: "
                f"{successful_results['latency_seconds'].mean():.4f} sec"
            )

            print(
                f"Average latency score: "
                f"{successful_results['latency_score'].mean():.4f}"
            )

            print(
                f"Average reward: "
                f"{successful_results['reward'].mean():.4f}"
            )

            print("\nAction distribution:")

            print(
                successful_results[
                    "chosen_action"
                ].value_counts()
            )

            print(
                "\nAverage reward by action:"
            )

            action_rewards = (
                successful_results
                .groupby(
                    "chosen_action"
                )["reward"]
                .agg(
                    ["count", "mean"]
                )
                .sort_values(
                    "mean",
                    ascending=False,
                )
            )

            print(action_rewards)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
    "\nCheckpoint 3.5 ground-truth "
    "evaluation complete."
)


if __name__ == "__main__":
    main()