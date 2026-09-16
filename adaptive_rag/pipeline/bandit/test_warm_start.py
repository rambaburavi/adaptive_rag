import os
import sys

import pandas as pd

from linucb import LinUCB
from warm_start import warm_start_from_results
from context import build_context

PIPELINE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROJECT_ROOT = os.path.dirname(
    PIPELINE_DIR
)

RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint3_reward_results.csv",
)


def main():

    print("=" * 70)
    print("CHECKPOINT 4.3 - WARM START TEST")
    print("=" * 70)

    print(
        "\nLoading Checkpoint 3.5 results..."
    )

    df = pd.read_csv(
        RESULTS_FILE
    )

    successful_df = df[
        df["status"] == "success"
    ].copy()

    print(
        f"Total results: "
        f"{len(df)}"
    )

    print(
        f"Successful results: "
        f"{len(successful_df)}"
    )

    print(
        "\nAction distribution:"
    )

    print(
        successful_df[
            "chosen_action"
        ].value_counts()
    )

    print(
        "\nInitializing warm-start LinUCB..."
    )

    bandit = warm_start_from_results(
        successful_df,
        alpha=1.0,
    )

    print(
        "\nWarm-start complete."
    )

    print(
        "\nLearned parameters:"
    )

    for parameter in bandit.get_parameters():

        arm = parameter["arm"]

        action = {
            0: "answer_directly",
            1: "retrieve",
            2: "decompose",
        }[arm]

        theta = parameter[
            "theta"
        ]

        print(
            f"\nArm {arm}: {action}"
        )

        print(
            "Theta:"
        )

        print(theta)

    test_features = {
        "query_length": 32,
        "num_words": 7,
        "has_compare": False,
        "has_relationship": True,
        "has_how": True,
        "is_definition": False,
        "num_concepts": 2,
        "complexity_score": 0.40,
    }

    test_context = build_context(
        test_features
    )

    scores = bandit.predict(
        test_context
    )

    selected_arm = bandit.select_arm(
        test_context
    )

    print(
        "\nTest context:"
    )

    print(test_context)

    print(
        "\nWarm-start UCB scores:"
    )

    print(scores)

    print(
        "\nSelected arm:"
    )

    print(
        selected_arm
    )

    print(
        "\nSelected action:"
    )

    print(
        {
            0: "answer_directly",
            1: "retrieve",
            2: "decompose",
        }[selected_arm]
    )

    print(
        "\nWarm-start test successful."
    )


if __name__ == "__main__":
    main()