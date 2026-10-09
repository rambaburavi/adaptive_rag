import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import chromadb
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# PROJECT IMPORTS
# ============================================================

from pipeline.naive_rag import retrieve_documents
from pipeline.feature_extractor import extract_features
from pipeline.adaptive_router import adaptive_route

from pipeline.actions.retrieve import retrieve
from pipeline.actions.decompose import decompose
from pipeline.actions.answer_directly import answer_directly

from pipeline.bandit.context import build_context
from pipeline.bandit.linucb import LinUCB

from pipeline.reward.reward_function import compute_reward


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

COLLECTION_NAME = "wikipedia_rag"

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "queries"
    / "full_references_frozen.csv"
)

QUERY_FILE = (
    PROJECT_ROOT
    / "data"
    / "queries"
    / "evaluation_queries.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoint5"
)

DETAIL_FILE = (
    RESULTS_DIR
    / "checkpoint5_results.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "checkpoint5_summary.csv"
)

CATEGORY_FILE = (
    RESULTS_DIR
    / "checkpoint5_category_summary.csv"
)

ACTION_FILE = (
    RESULTS_DIR
    / "checkpoint5_action_summary.csv"
)

LEARNING_FILE = (
    RESULTS_DIR
    / "checkpoint5_linucb_learning.csv"
)

LINUCB_STATE_FILE = (
    RESULTS_DIR
    / "linucb_state.npz"
)

# Save state frequently so a rate limit / crash loses
# as little work as possible.
SAVE_EVERY_QUERY = 1


# ============================================================
# ACTION MAPPING
# ============================================================

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


# ============================================================
# SETUP
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("CHECKPOINT 5 — FULL EMPIRICAL EVALUATION")
print("RESUMABLE ONLINE LINUCB EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

queries_df = pd.read_csv(
    QUERY_FILE
)

references_df = pd.read_csv(
    REFERENCE_FILE
)

print(
    f"\nEvaluation queries : {len(queries_df)}"
)

print(
    f"Frozen references  : {len(references_df)}"
)


if len(queries_df) != 200:
    raise ValueError(
        f"Expected 200 evaluation queries, "
        f"found {len(queries_df)}"
    )


if len(references_df) != 200:
    raise ValueError(
        f"Expected 200 frozen references, "
        f"found {len(references_df)}"
    )


# ============================================================
# VALIDATE REFERENCE SET
# ============================================================

if references_df["query"].duplicated().any():
    raise ValueError(
        "Frozen reference file contains duplicate queries."
    )


reference_map = dict(
    zip(
        references_df["query"],
        references_df["reference_answer"],
    )
)


missing_refs = [
    query
    for query in queries_df["query"]
    if query not in reference_map
]


if missing_refs:
    raise ValueError(
        f"Missing frozen references for "
        f"{len(missing_refs)} queries."
    )


empty_refs = [
    query
    for query, reference in reference_map.items()
    if pd.isna(reference)
    or str(reference).strip() == ""
]


if empty_refs:
    raise ValueError(
        f"Found {len(empty_refs)} empty frozen references."
    )


print(
    "Frozen reference validation: PASSED"
)


# ============================================================
# LOAD CHROMA
# ============================================================

print(
    "\nLoading ChromaDB..."
)

chroma_client = chromadb.PersistentClient(
    path=str(
        PROJECT_ROOT
        / "chroma_db"
    )
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Chroma collection loaded: "
    f"{COLLECTION_NAME}"
)


# ============================================================
# LOAD GROQ
# ============================================================

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


groq_client = Groq(
    api_key=groq_api_key
)

print(
    "Groq client loaded."
)


# ============================================================
# HELPER
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# SYSTEM 1 — NAIVE RAG
# ============================================================

def run_naive_rag(query):

    start = time.perf_counter()

    documents, metadatas, distances = (
        retrieve_documents(
            query,
            top_k=5,
        )
    )

    evidence = []

    for text, meta, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        evidence.append(
            {
                "text": text,
                "source": meta.get(
                    "source",
                    "unknown",
                ),
                "score": 1 - distance,
            }
        )

    context = "\n\n".join(
        f"[{i + 1}] {item['text']}"
        for i, item in enumerate(evidence)
    )

    prompt = (
        "Answer the question using the context below. "
        "If the context does not fully answer the question, "
        "say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "action": "retrieve",
        "answer": answer,
        "elapsed_time": elapsed,
        "evidence": evidence,
    }


# ============================================================
# SYSTEM 2 — MANUAL ADAPTIVE ROUTER
# ============================================================

def run_manual_router(query):

    start = time.perf_counter()

    action, features = adaptive_route(
        query
    )

    state = {
        "evidence": [],
        "actions_taken": [],
        "sub_answers": [],
    }

    if action == "answer_directly":

        state, answer = answer_directly(
            query,
            collection,
            groq_client,
            state,
        )

    elif action == "retrieve":

        state, answer = retrieve(
            query,
            collection,
            groq_client,
            state,
        )

    elif action == "decompose":

        state, answer = decompose(
            query,
            collection,
            groq_client,
            state,
        )

    else:

        raise ValueError(
            f"Unknown manual-router action: {action}"
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "action": action,
        "answer": normalize_text(answer),
        "elapsed_time": elapsed,
        "evidence": state.get(
            "evidence",
            [],
        ),
    }


# ============================================================
# LINUCB CREATION
# ============================================================

linucb = LinUCB(
    n_arms=3,
    context_dim=8,
    alpha=1.0,
)


# ============================================================
# LINUCB STATE SAVE
# ============================================================

def save_linucb_state():

    np.savez(
        LINUCB_STATE_FILE,
        A=np.array(
            linucb.A
        ),
        b=np.array(
            linucb.b
        ),
        alpha=np.array(
            [linucb.alpha]
        ),
        n_arms=np.array(
            [linucb.n_arms]
        ),
        context_dim=np.array(
            [linucb.context_dim]
        ),
    )


# ============================================================
# LINUCB STATE LOAD
# ============================================================

def load_linucb_state():

    if not LINUCB_STATE_FILE.exists():

        print(
            "\nNo saved LinUCB state found."
        )

        print(
            "Starting from CP4 warm-start state."
        )

        return False

    data = np.load(
        LINUCB_STATE_FILE
    )

    saved_A = data["A"]
    saved_b = data["b"]

    if saved_A.shape != (
        3,
        8,
        8,
    ):

        raise ValueError(
            "Saved LinUCB A matrix has "
            f"unexpected shape: {saved_A.shape}"
        )

    if saved_b.shape != (
        3,
        8,
    ):

        raise ValueError(
            "Saved LinUCB b matrix has "
            f"unexpected shape: {saved_b.shape}"
        )

    linucb.A = [
        saved_A[i].copy()
        for i in range(3)
    ]

    linucb.b = [
        saved_b[i].copy()
        for i in range(3)
    ]

    print(
        "\nExisting LinUCB state restored."
    )

    return True


# ============================================================
# CP4 WARM START
# ============================================================

def warm_start_linucb():

    candidates = [
        (
            PROJECT_ROOT
            / "eval"
            / "checkpoint4_linucb_results.csv"
        ),
        (
            PROJECT_ROOT
            / "eval"
            / "checkpoint3_reward_results.csv"
        ),
    ]

    warm_file = None

    for path in candidates:

        if path.exists():

            warm_file = path

            break

    if warm_file is None:

        print(
            "\nNo CP4 warm-start file found."
        )

        return

    df = pd.read_csv(
        warm_file
    )

    print(
        f"\nWarm-start source: "
        f"{warm_file.name}"
    )

    action_column = None
    reward_column = None

    for col in [
    "action",
    "selected_action",
    "action_name",
    "chosen_action",
]:

        if col in df.columns:

            action_column = col

            break

    for col in [
        "reward",
        "final_reward",
    ]:

        if col in df.columns:

            reward_column = col

            break

    if (
        action_column is None
        or reward_column is None
    ):

        print(
            "Could not identify CP4 "
            "action/reward columns."
        )

        return

    updated = 0

    for _, row in df.iterrows():

        action = normalize_text(
            row[action_column]
        )

        if action not in ACTION_TO_ARM:
            continue

        try:

            reward = float(
                row[reward_column]
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        query = normalize_text(
            row.get(
                "query",
                "",
            )
        )

        if not query:
            continue

        features = extract_features(
            query
        )

        context = build_context(features)

        arm = ACTION_TO_ARM[
            action
        ]

        linucb.update(
            arm,
            context,
            reward,
        )

        updated += 1

    print(
        f"CP4 warm-start observations: "
        f"{updated}"
    )


# ============================================================
# RESTORE OR WARM START
# ============================================================

if not load_linucb_state():

    warm_start_linucb()

    save_linucb_state()

    print(
        "Initial LinUCB state saved."
    )


# ============================================================
# SYSTEM 3 — ONLINE LINUCB
# ============================================================

def run_linucb(query):

    start = time.perf_counter()

    features = extract_features(
        query
    )

    context = build_context(
        features
    )

    ucb_scores = linucb.predict(
        context
    )

    selected_arm = int(
        np.argmax(
            ucb_scores
        )
    )

    action = ARM_TO_ACTION[
        selected_arm
    ]

    state = {
        "evidence": [],
        "actions_taken": [],
        "sub_answers": [],
    }

    if action == "answer_directly":

        state, answer = answer_directly(
            query,
            collection,
            groq_client,
            state,
        )

    elif action == "retrieve":

        state, answer = retrieve(
            query,
            collection,
            groq_client,
            state,
        )

    elif action == "decompose":

        state, answer = decompose(
            query,
            collection,
            groq_client,
            state,
        )

    else:

        raise ValueError(
            f"Unknown LinUCB action: {action}"
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "action": action,
        "answer": normalize_text(answer),
        "elapsed_time": elapsed,
        "evidence": state.get(
            "evidence",
            [],
        ),
        "features": features,
        "context": context,
        "ucb_scores": (
            ucb_scores.tolist()
        ),
    }


# ============================================================
# REWARD
# ============================================================

def calculate_reward(
    query,
    answer,
    evidence,
    elapsed_time,
    action,
):

    reference = reference_map[
        query
    ]

    return compute_reward(
        query=query,
        answer=answer,
        evidence_chunks=evidence,
        latency_seconds=elapsed_time,
        action_name=action,
        reference_text=reference,
    )


# ============================================================
# RESULT FILE
# ============================================================

def load_existing_results():

    if not DETAIL_FILE.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            DETAIL_FILE
        )

        print(
            f"\nExisting result rows: "
            f"{len(df)}"
        )

        return df

    except Exception as exc:

        print(
            f"\nCould not read existing "
            f"results: {exc}"
        )

        return pd.DataFrame()


existing_results = (
    load_existing_results()
)


# ============================================================
# COMPLETION TRACKING
# ============================================================

def completed_pairs(df):

    completed = set()

    if df.empty:
        return completed

    required_columns = {
        "query_number",
        "system",
    }

    if not required_columns.issubset(
        df.columns
    ):

        return completed

    for _, row in df.iterrows():

        action = normalize_text(
            row.get(
                "action",
                "",
            )
        )

        # Never treat an ERROR row as completed.
        if action == "ERROR":
            continue

        try:

            query_number = int(
                row["query_number"]
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        system = normalize_text(
            row["system"]
        )

        completed.add(
            (
                query_number,
                system,
            )
        )

    return completed


completed = completed_pairs(
    existing_results
)


print(
    f"Completed system/query pairs: "
    f"{len(completed)} / 600"
)


# ============================================================
# ALL RESULTS
# ============================================================

all_rows = []

if not existing_results.empty:

    all_rows.extend(
        existing_results.to_dict(
            orient="records"
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results():

    if not all_rows:
        return

    df = pd.DataFrame(
        all_rows
    )

    # Remove accidental duplicate
    # system/query pairs.
    if {
        "query_number",
        "system",
    }.issubset(df.columns):

        df = (
            df
            .drop_duplicates(
                subset=[
                    "query_number",
                    "system",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "query_number",
                    "system",
                ]
            )
        )

    df.to_csv(
        DETAIL_FILE,
        index=False,
    )


# ============================================================
# LINUCB UPDATE
# ============================================================

def update_linucb(
    query,
    action,
    reward,
):

    features = extract_features(
        query
    )

    context = build_context(
        features
    )

    arm = ACTION_TO_ARM[
        action
    ]

    linucb.update(
        arm,
        context,
        float(reward),
    )

    save_linucb_state()


# ============================================================
# GROQ RATE-LIMIT DETECTION
# ============================================================

def is_rate_limit_error(exc):

    text = str(exc).lower()

    return (
        "429" in text
        or "rate limit" in text
        or "tokens per day" in text
        or "tpd" in text
    )


# ============================================================
# SYSTEM EXECUTION
# ============================================================

SYSTEMS = [
    (
        "Naive RAG",
        run_naive_rag,
    ),
    (
        "Manual Router",
        run_manual_router,
    ),
    (
        "LinUCB",
        run_linucb,
    ),
]


# ============================================================
# EXPERIMENT
# ============================================================

experiment_stopped = False


try:

    for index, query_row in queries_df.iterrows():

        query_number = (
            index + 1
        )

        query = normalize_text(
            query_row["query"]
        )

        category = normalize_text(
            query_row[
                "expected_difficulty_label"
            ]
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Query {query_number}/200"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {query}"
        )

        print(
            "=" * 70
        )


        for system_name, runner in SYSTEMS:

            pair = (
                query_number,
                system_name,
            )

            # ------------------------------------------------
            # SKIP ALREADY COMPLETED SYSTEM
            # ------------------------------------------------

            if pair in completed:

                print(
                    f"\nSKIP {system_name}"
                    f" — already completed."
                )

                continue


            print(
                f"\nRunning {system_name}..."
            )


            try:

                result = runner(
                    query
                )


                reward_info = calculate_reward(
                    query=query,
                    answer=result[
                        "answer"
                    ],
                    evidence=result[
                        "evidence"
                    ],
                    elapsed_time=result[
                        "elapsed_time"
                    ],
                    action=result[
                        "action"
                    ],
                )


                reward = reward_info.get(
    "reward",
    np.nan,
)


                row_data = {

                    "query_number":
                        query_number,

                    "query":
                        query,

                    "category":
                        category,

                    "system":
                        system_name,

                    "action":
                        result[
                            "action"
                        ],

                    "answer":
                        result[
                            "answer"
                        ],

                    "reference_answer":
                        reference_map[
                            query
                        ],

                    "latency_seconds":
                        result[
                            "elapsed_time"
                        ],

                    "answer_quality":
                        reward_info.get(
                            "answer_quality",
                            np.nan,
                        ),

                    "faithfulness":
                        reward_info.get(
                            "faithfulness",
                            np.nan,
                        ),

                    "retrieval_relevance":
                        reward_info.get(
                            "retrieval_precision",
                            np.nan,
                        ),

                    "latency_score":
    reward_info.get(
        "latency_score",
        np.nan,
    ),

                    "reward":
                        reward,
                }


                # --------------------------------------------
                # LINUCB METADATA
                # --------------------------------------------

                if system_name == "LinUCB":

                    features = result.get(
                        "features",
                        {},
                    )

                    ucb_scores = result.get(
                        "ucb_scores",
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                    )

                    row_data.update(
                        {

                            "query_length":
                                features.get(
                                    "query_length",
                                    np.nan,
                                ),

                            "num_words":
                                features.get(
                                    "num_words",
                                    np.nan,
                                ),

                            "has_compare":
                                features.get(
                                    "has_compare",
                                    np.nan,
                                ),

                            "has_relationship":
                                features.get(
                                    "has_relationship",
                                    np.nan,
                                ),

                            "has_how":
                                features.get(
                                    "has_how",
                                    np.nan,
                                ),

                            "is_definition":
                                features.get(
                                    "is_definition",
                                    np.nan,
                                ),

                            "num_concepts":
                                features.get(
                                    "num_concepts",
                                    np.nan,
                                ),

                            "complexity_score":
                                features.get(
                                    "complexity_score",
                                    np.nan,
                                ),

                            "ucb_answer_directly":
                                ucb_scores[0],

                            "ucb_retrieve":
                                ucb_scores[1],

                            "ucb_decompose":
                                ucb_scores[2],
                        }
                    )


                # --------------------------------------------
                # SAVE RESULT IMMEDIATELY
                # --------------------------------------------

                all_rows.append(
                    row_data
                )

                completed.add(
                    pair
                )

                save_results()


                print(
                    f"{system_name}: "
                    f"action={result['action']} "
                    f"reward={reward:.4f} "
                    f"latency="
                    f"{result['elapsed_time']:.2f}s"
                )


                # --------------------------------------------
                # ONLY LINUCB LEARNS
                # --------------------------------------------

                if system_name == "LinUCB":

                    update_linucb(
                        query=query,
                        action=result[
                            "action"
                        ],
                        reward=reward,
                    )

                    print(
                        "LinUCB state updated "
                        "and saved."
                    )


            except Exception as exc:

                print(
                    f"\nERROR in "
                    f"{system_name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )


                # --------------------------------------------
                # RATE LIMIT
                # --------------------------------------------

                if is_rate_limit_error(
                    exc
                ):

                    print(
                        "\n"
                        + "!" * 70
                    )

                    print(
                        "GROQ RATE LIMIT DETECTED."
                    )

                    print(
                        "Stopping CP5 safely."
                    )

                    print(
                        "Current results and "
                        "LinUCB state have "
                        "already been saved."
                    )

                    print(
                        "Run the same command "
                        "again later to resume."
                    )

                    print(
                        "!" * 70
                    )

                    experiment_stopped = True

                    raise


                # --------------------------------------------
                # OTHER ERROR
                # --------------------------------------------

                print(
                    "Non-rate-limit error."
                )

                print(
                    "The system/query pair "
                    "will remain incomplete "
                    "and will be retried "
                    "on the next run."
                )


        # ----------------------------------------------------
        # QUERY CHECKPOINT
        # ----------------------------------------------------

        if (
            query_number
            % SAVE_EVERY_QUERY
            == 0
        ):

            save_results()

            save_linucb_state()

            completed = completed_pairs(
                pd.DataFrame(
                    all_rows
                )
            )

            print(
                "\nCHECKPOINT SAVED"
            )

            print(
                f"Completed pairs: "
                f"{len(completed)}/600"
            )


except Exception as exc:

    if is_rate_limit_error(exc):

        # Clean experiment pause.
        pass

    else:

        print(
            "\nExperiment stopped because "
            f"of: {type(exc).__name__}: {exc}"
        )

        print(
            "Progress has been saved."
        )


# ============================================================
# AGGREGATION
# ============================================================

results_df = pd.DataFrame(
    all_rows
)


if results_df.empty:

    print(
        "\nNo results available yet."
    )

    sys.exit(0)


# ============================================================
# FINAL DEDUPLICATION
# ============================================================

if {
    "query_number",
    "system",
}.issubset(
    results_df.columns
):

    results_df = (
        results_df
        .drop_duplicates(
            subset=[
                "query_number",
                "system",
            ],
            keep="last",
        )
        .sort_values(
            [
                "query_number",
                "system",
            ]
        )
    )


results_df.to_csv(
    DETAIL_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

numeric_columns = [
    "reward",
    "answer_quality",
    "faithfulness",
    "retrieval_relevance",
    "latency_seconds",
    "latency_score",
]


available_numeric = [
    col
    for col in numeric_columns
    if col in results_df.columns
]


summary_df = (
    results_df
    .groupby("system")[
        available_numeric
    ]
    .agg(
        [
            "mean",
            "std",
            "count",
        ]
    )
)

summary_df.to_csv(
    SUMMARY_FILE
)


# ============================================================
# CATEGORY SUMMARY
# ============================================================

category_summary = (
    results_df
    .groupby(
        [
            "system",
            "category",
        ]
    )[available_numeric]
    .mean()
    .reset_index()
)

category_summary.to_csv(
    CATEGORY_FILE,
    index=False,
)


# ============================================================
# ACTION SUMMARY
# ============================================================

action_summary = (
    results_df
    .groupby(
        [
            "system",
            "action",
        ]
    )
    .agg(
        count=(
            "action",
            "size",
        ),
        avg_reward=(
            "reward",
            "mean",
        ),
        avg_latency=(
            "latency_seconds",
            "mean",
        ),
        avg_answer_quality=(
            "answer_quality",
            "mean",
        ),
    )
    .reset_index()
)

action_summary.to_csv(
    ACTION_FILE,
    index=False,
)


# ============================================================
# LINUCB LEARNING CURVE
# ============================================================

linucb_results = (
    results_df[
        results_df["system"]
        == "LinUCB"
    ]
    .sort_values(
        "query_number"
    )
    .copy()
)


if not linucb_results.empty:

    linucb_results[
        "cumulative_reward"
    ] = (
        linucb_results[
            "reward"
        ]
        .fillna(0)
        .cumsum()
    )

    linucb_results[
        "rolling_reward_20"
    ] = (
        linucb_results[
            "reward"
        ]
        .rolling(
            window=20,
            min_periods=1,
        )
        .mean()
    )

    linucb_results[
        [
            "query_number",
            "query",
            "category",
            "action",
            "reward",
            "cumulative_reward",
            "rolling_reward_20",
        ]
    ].to_csv(
        LEARNING_FILE,
        index=False,
    )


# ============================================================
# CURRENT STATUS
# ============================================================

completed_now = completed_pairs(
    results_df
)

successful_pairs = len(
    completed_now
)

print(
    "\n"
    + "=" * 70
)

if successful_pairs >= 600:

    print(
        "CP5 — ALL 600 SYSTEM/QUERY "
        "EVALUATIONS COMPLETE"
    )

else:

    print(
        "CP5 — EXPERIMENT PAUSED / "
        "PARTIALLY COMPLETE"
    )


print(
    "=" * 70
)

print(
    f"Completed pairs: "
    f"{successful_pairs}/600"
)

print(
    f"Remaining pairs: "
    f"{600 - successful_pairs}"
)


# ============================================================
# CURRENT SYSTEM SUMMARY
# ============================================================

print(
    "\nCURRENT SYSTEM SUMMARY:"
)

print(
    results_df
    .groupby("system")[
        available_numeric
    ]
    .mean()
    .round(4)
)


# ============================================================
# CURRENT ACTION DISTRIBUTION
# ============================================================

print(
    "\nCURRENT ACTION DISTRIBUTION:"
)

print(
    pd.crosstab(
        results_df["system"],
        results_df["action"],
    )
)


# ============================================================
# CURRENT CATEGORY REWARD
# ============================================================

print(
    "\nCURRENT CATEGORY REWARD:"
)

print(
    results_df
    .groupby(
        [
            "system",
            "category",
        ]
    )["reward"]
    .mean()
    .round(4)
)


# ============================================================
# OUTPUT FILES
# ============================================================

print(
    "\nOutput directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nResult file:"
)

print(
    DETAIL_FILE
)

print(
    "\nLinUCB state:"
)

print(
    LINUCB_STATE_FILE
)

print(
    "\nSummary files:"
)

print(
    SUMMARY_FILE
)

print(
    CATEGORY_FILE
)

print(
    ACTION_FILE
)

print(
    LEARNING_FILE
)


if experiment_stopped:

    print(
        "\nCP5 is paused safely."
    )

    print(
        "Run the same command again "
        "to continue."
    )
