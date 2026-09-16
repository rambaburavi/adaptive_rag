import os
import sys

# ---------------------------------------------------------
# Allow imports from pipeline/
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")

sys.path.append(PIPELINE_DIR)

from reward.faithfulness import score_faithfulness
from reward.reward_function import compute_reward


# ---------------------------------------------------------
# Helper: create evidence
# ---------------------------------------------------------

def make_evidence(text, score):
    return [
        {
            "text": text,
            "source": "test_source",
            "score": score,
        }
    ]


# ---------------------------------------------------------
# Run one test
# ---------------------------------------------------------

def run_test(
    test_name,
    query,
    answer,
    evidence,
    latency,
):
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"Latency: {latency:.2f} seconds")

    # -----------------------------------------------------
    # NLI DEBUG
    # -----------------------------------------------------

    print("\nNLI DEBUG:")

    debug_faithfulness = score_faithfulness(
        answer,
        evidence,
        debug=True,
    )

    print(
        f"\nDebug faithfulness score: "
        f"{debug_faithfulness:.4f}"
    )

    # -----------------------------------------------------
    # Compute complete reward
    # -----------------------------------------------------

    result = compute_reward(
    query=query,
    answer=answer,
    evidence_chunks=evidence,
    latency_seconds=latency,
    action_name="retrieve",
)
    # -----------------------------------------------------
    # Display scores
    # -----------------------------------------------------

    print("\nScores:")

    print(
        f"  Faithfulness       : "
        f"{result['faithfulness']:.4f}"
    )

    print(
        f"  Retrieval precision: "
        f"{result['retrieval_precision']:.4f}"
    )

    print(
        f"  Latency score      : "
        f"{result['latency_score']:.4f}"
    )

    print(
        f"  FINAL REWARD       : "
        f"{result['reward']:.4f}"
    )

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    tests = [

        # =================================================
        # GOOD EXAMPLE 1
        # =================================================

        {
            "name": "Good - strongly supported answer",

            "query": "What is machine learning?",

            "answer": (
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn patterns from data."
            ),

            "evidence": make_evidence(
                (
                    "Machine learning is a subset of artificial intelligence "
                    "that enables computer systems to learn from data and "
                    "improve their performance."
                ),
                0.92,
            ),

            "latency": 1.5,
        },


        # =================================================
        # GOOD EXAMPLE 2
        # =================================================

        {
            "name": "Good - relevant retrieval and fast response",

            "query": "What is deep learning?",

            "answer": (
                "Deep learning is a type of machine learning that uses "
                "multi-layer neural networks."
            ),

            "evidence": make_evidence(
                (
                    "Deep learning is a branch of machine learning based "
                    "on artificial neural networks with multiple layers."
                ),
                0.90,
            ),

            "latency": 1.8,
        },


        # =================================================
        # GOOD EXAMPLE 3
        # =================================================

        {
            "name": "Good - multiple relevant chunks",

            "query": (
                "How are neural networks related to deep learning?"
            ),

            "answer": (
                "Deep learning commonly uses neural networks with multiple "
                "layers to learn increasingly complex representations."
            ),

            "evidence": [
                {
                    "text": (
                        "Deep learning uses neural networks containing "
                        "multiple layers."
                    ),
                    "source": "deep_learning",
                    "score": 0.91,
                },
                {
                    "text": (
                        "Neural networks are computational models made "
                        "of interconnected artificial neurons."
                    ),
                    "source": "neural_network",
                    "score": 0.87,
                },
            ],

            "latency": 2.2,
        },


        # =================================================
        # BAD EXAMPLE 1
        # =================================================

        {
            "name": "Bad - unsupported answer",

            "query": "What is machine learning?",

            "answer": (
                "Machine learning was invented in 1956 and was originally "
                "created by NASA for space exploration."
            ),

            "evidence": make_evidence(
                (
                    "Machine learning enables computer systems to learn "
                    "patterns from data."
                ),
                0.90,
            ),

            "latency": 2.0,
        },


        # =================================================
        # BAD EXAMPLE 2
        # =================================================

        {
            "name": "Bad - poor retrieval",

            "query": "What is deep learning?",

            "answer": (
                "Deep learning is a machine learning approach based on "
                "multi-layer neural networks."
            ),

            "evidence": make_evidence(
                (
                    "Computer science studies computation, algorithms, "
                    "and information processing."
                ),
                0.15,
            ),

            "latency": 2.0,
        },


        # =================================================
        # BAD EXAMPLE 3
        # =================================================

        {
            "name": "Bad - very slow response",

            "query": "What is artificial intelligence?",

            "answer": (
                "Artificial intelligence refers to computational systems "
                "designed to perform tasks associated with human "
                "intelligence."
            ),

            "evidence": make_evidence(
                (
                    "Artificial intelligence is the field of creating "
                    "systems capable of performing tasks associated with "
                    "human intelligence."
                ),
                0.93,
            ),

            "latency": 14.0,
        },


        # =================================================
        # BAD EXAMPLE 4
        # =================================================

        {
            "name": (
                "Bad - unsupported answer and poor retrieval"
            ),

            "query": "What is a neural network?",

            "answer": (
                "A neural network is a database system designed to store "
                "large amounts of relational data."
            ),

            "evidence": make_evidence(
                (
                    "Machine learning algorithms learn patterns from "
                    "training data."
                ),
                0.10,
            ),

            "latency": 12.0,
        },
    ]


    # -----------------------------------------------------
    # Run all tests
    # -----------------------------------------------------

    results = []

    for test in tests:

        result = run_test(
            test_name=test["name"],
            query=test["query"],
            answer=test["answer"],
            evidence=test["evidence"],
            latency=test["latency"],
        )

        results.append(result)


    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n\n")

    print("=" * 70)
    print("CHECKPOINT 3 SANITY SUMMARY")
    print("=" * 70)

    for result in results:

        print(
            f"{result['query'][:45]:45} "
            f"-> reward={result['reward']:.4f}"
        )

    print("\nCheckpoint 3 sanity test complete.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()