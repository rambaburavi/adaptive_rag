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

sys.path.append(PIPELINE_DIR)

from actions import ACTIONS
from actions.common import new_state
from feature_extractor import extract_features
from bandit.context import build_context
from bandit.linucb import LinUCB
from bandit.warm_start import warm_start_from_results
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

WARM_START_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint3_reward_results.csv",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint4_linucb_results.csv",
)

QUERIES_PER_CATEGORY = 5

ACTION_TO_ARM = {
    "answer_directly": 0,
    "retrieve": 1,
    "decompose": 2,
}

ARM_TO_ACTION = {
    0: "answer_directly",
    1: "retrieve",
    2: "decompose",
}


def load_ground_truth():

    reference_df = pd.read_csv(
        REFERENCE_FILE
    )

    required_columns = {
        "query",
        "reference_answer",
    }

    missing_columns = (
        required_columns
        - set(reference_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing reference columns: "
            f"{missing_columns}"
        )

    return dict(
        zip(
            reference_df["query"]
            .astype(str)
            .str.strip(),

            reference_df[
                "reference_answer"
            ].astype(str),
        )
    )


def load_queries():

    df = pd.read_csv(
        QUERY_FILE
    )

    evaluation_df = (
        df.groupby(
            "expected_difficulty_label",
            sort=False,
        )
        .head(
            QUERIES_PER_CATEGORY
        )
        .reset_index(drop=True)
    )

    return evaluation_df


def load_chroma():

    print(
        "\nLoading ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection loaded: "
        f"{COLLECTION_NAME}"
    )

    return collection


def load_groq():

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Check your .env file."
        )

    print(
        "\nConnecting to Groq..."
    )

    client = Groq(
        api_key=groq_api_key
    )

    print(
        "Groq connection ready."
    )

    return client


def run_action(
    action_name,
    query,
    collection,
    groq_client,
):

    action_fn = ACTIONS[
        action_name
    ]

    state = new_state()

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

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        print(
            f"ERROR: {error}"
        )

        return {
            "state": state,
            "answer": "",
            "latency_seconds":
                latency_seconds,
            "status": "action_error",
        }

    latency_seconds = (
        time.perf_counter()
        - start_time
    )

    return {
        "state": state,
        "answer": answer,
        "latency_seconds":
            latency_seconds,
        "status": status,
    }


def main():

    print("=" * 70)
    print(
        "CHECKPOINT 4.4 - ONLINE LINUCB"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load resources
    # --------------------------------------------------

    collection = load_chroma()

    groq_client = load_groq()

    print(
        "\nLoading evaluation queries..."
    )

    evaluation_df = load_queries()

    print(
        f"Queries selected: "
        f"{len(evaluation_df)}"
    )

    print(
        "\nDistribution:"
    )

    print(
        evaluation_df[
            "expected_difficulty_label"
        ].value_counts()
    )

    print(
        "\nLoading ground-truth references..."
    )

    reference_map = (
        load_ground_truth()
    )

    print(
        f"Ground-truth references loaded: "
        f"{len(reference_map)}"
    )

    # --------------------------------------------------
    # Warm start
    # --------------------------------------------------

    print(
        "\nLoading warm-start observations..."
    )

    warm_start_df = pd.read_csv(
        WARM_START_FILE
    )

    warm_start_df = warm_start_df[
        warm_start_df["status"]
        == "success"
    ].copy()

    print(
        f"Warm-start observations: "
        f"{len(warm_start_df)}"
    )

    bandit = warm_start_from_results(
        warm_start_df,
        alpha=1.0,
    )

    print(
        "\nWarm-start LinUCB initialized."
    )

    # --------------------------------------------------
    # Online learning
    # --------------------------------------------------

    results = []

    cumulative_reward = 0.0

    for index, row in evaluation_df.iterrows():

        query = str(
            row["query"]
        ).strip()

        expected_label = str(
            row[
                "expected_difficulty_label"
            ]
        )

        expected_sources = str(
            row.get(
                "expected_sources",
                "",
            )
        )

        reference_answer = (
            reference_map.get(
                query,
                "",
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            f"QUERY "
            f"{index + 1}/"
            f"{len(evaluation_df)}"
        )

        print(
            "=" * 70
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Expected category: "
            f"{expected_label}"
        )

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------

        features = extract_features(
            query
        )

        context = build_context(
            features
        )

        # --------------------------------------------------
        # LinUCB selection
        # --------------------------------------------------

        ucb_scores = (
            bandit.predict(
                context
            )
        )

        selected_arm = (
            bandit.select_arm(
                context
            )
        )

        action_name = (
            ARM_TO_ACTION[
                selected_arm
            ]
        )

        print(
            "\nContext:"
        )

        print(context)

        print(
            "\nUCB scores:"
        )

        print(ucb_scores)

        print(
            f"\nSelected arm: "
            f"{selected_arm}"
        )

        print(
            f"Selected action: "
            f"{action_name}"
        )

        # --------------------------------------------------
        # Execute action
        # --------------------------------------------------

        print(
            "\nExecuting action..."
        )

        action_result = run_action(
            action_name,
            query,
            collection,
            groq_client,
        )

        answer = (
            action_result["answer"]
        )

        latency_seconds = (
            action_result[
                "latency_seconds"
            ]
        )

        status = (
            action_result["status"]
        )

        if status != "success":

            reward = 0.0

            bandit.update(
                arm=selected_arm,
                context=context,
                reward=reward,
            )

            results.append({
                "query": query,
                "expected_label":
                    expected_label,
                "expected_sources":
                    expected_sources,
                "selected_arm":
                    selected_arm,
                "chosen_action":
                    action_name,
                "complexity_score":
                    features[
                        "complexity_score"
                    ],
                "answer": "",
                "answer_quality": 0.0,
                "faithfulness": 0.0,
                "retrieval_precision":
                    None,
                "latency_seconds":
                    round(
                        latency_seconds,
                        4,
                    ),
                "latency_score": 0.0,
                "reward": 0.0,
                "cumulative_reward":
                    round(
                        cumulative_reward,
                        4,
                    ),
                "status": status,
            })

            continue

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        evidence = (
            action_result[
                "state"
            ].get(
                "evidence",
                [],
            )
        )

        reward_result = (
            compute_reward(
                query=query,
                answer=answer,
                evidence_chunks=evidence,
                latency_seconds=
                    latency_seconds,
                action_name=
                    action_name,
                reference_text=
                    reference_answer,
            )
        )

        reward = float(
            reward_result[
                "reward"
            ]
        )

        # --------------------------------------------------
        # UPDATE BANDIT
        # --------------------------------------------------

        bandit.update(
            arm=selected_arm,
            context=context,
            reward=reward,
        )

        cumulative_reward += reward

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print(
            f"\nLatency: "
            f"{latency_seconds:.2f} sec"
        )

        print(
            f"Answer quality: "
            f"{reward_result['answer_quality']:.4f}"
        )

        print(
            f"Faithfulness: "
            f"{reward_result['faithfulness']}"
        )

        print(
            f"Retrieval relevance: "
            f"{reward_result['retrieval_precision']}"
        )

        print(
            f"Latency score: "
            f"{reward_result['latency_score']:.4f}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )

        print(
            f"Cumulative reward: "
            f"{cumulative_reward:.4f}"
        )

        # --------------------------------------------------
        # Save observation
        # --------------------------------------------------

        results.append({

            "query": query,

            "expected_label":
                expected_label,

            "expected_sources":
                expected_sources,

            "selected_arm":
                selected_arm,

            "chosen_action":
                action_name,

            "complexity_score":
                features[
                    "complexity_score"
                ],

            "answer":
                answer,

            "num_evidence":
                len(evidence),

            "answer_quality":
                reward_result[
                    "answer_quality"
                ],

            "faithfulness":
                reward_result[
                    "faithfulness"
                ],

            "retrieval_precision":
                reward_result[
                    "retrieval_precision"
                ],

            "latency_seconds":
                reward_result[
                    "latency_seconds"
                ],

            "latency_score":
                reward_result[
                    "latency_score"
                ],

            "reward":
                reward,

            "cumulative_reward":
                round(
                    cumulative_reward,
                    4,
                ),

            "status":
                status,
        })

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    successful_results = (
        results_df[
            results_df["status"]
            == "success"
        ]
    )

    print("\n\n")

    print("=" * 70)

    print(
        "CHECKPOINT 4.4 "
        "LINUCB SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nTotal queries: "
        f"{len(results_df)}"
    )

    print(
        f"Successful queries: "
        f"{len(successful_results)}"
    )

    if not successful_results.empty:

        print(
            f"\nAverage reward: "
            f"{successful_results['reward'].mean():.4f}"
        )

        print(
            f"Total cumulative reward: "
            f"{successful_results['reward'].sum():.4f}"
        )

        print(
            f"Average answer quality: "
            f"{successful_results['answer_quality'].mean():.4f}"
        )

        print(
            f"Average faithfulness: "
            f"{successful_results['faithfulness'].dropna().mean():.4f}"
        )

        print(
            f"Average retrieval relevance: "
            f"{successful_results['retrieval_precision'].dropna().mean():.4f}"
        )

        print(
            f"Average latency: "
            f"{successful_results['latency_seconds'].mean():.4f} sec"
        )

        print(
            "\nAction distribution:"
        )

        print(
            successful_results[
                "chosen_action"
            ].value_counts()
        )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nCheckpoint 4.4 complete."
    )


if __name__ == "__main__":
    main()