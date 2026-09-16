import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint4_linucb_results.csv",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "checkpoint4_6_learning_analysis.csv",
)


def main():

    print("=" * 70)
    print("CHECKPOINT 4.6 - LEARNING ANALYSIS")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Results file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    successful = df[
        df["status"] == "success"
    ].copy()

    if successful.empty:
        raise ValueError(
            "No successful results found."
        )

    # --------------------------------------------------
    # Query position
    # --------------------------------------------------

    successful["query_number"] = range(
        1,
        len(successful) + 1
    )

    # --------------------------------------------------
    # Cumulative statistics
    # --------------------------------------------------

    successful["cumulative_reward"] = (
        successful["reward"].cumsum()
    )

    successful["cumulative_average_reward"] = (
        successful["cumulative_reward"]
        / successful["query_number"]
    )

    # --------------------------------------------------
    # First half / second half
    # --------------------------------------------------

    midpoint = len(successful) // 2

    first_half = successful.iloc[
        :midpoint
    ]

    second_half = successful.iloc[
        midpoint:
    ]

    first_reward = (
        first_half["reward"].mean()
    )

    second_reward = (
        second_half["reward"].mean()
    )

    reward_change = (
        second_reward - first_reward
    )

    # --------------------------------------------------
    # Action statistics
    # --------------------------------------------------

    action_stats = (
        successful
        .groupby("chosen_action")
        .agg(
            observations=(
                "reward",
                "count",
            ),
            mean_reward=(
                "reward",
                "mean",
            ),
            mean_latency=(
                "latency_seconds",
                "mean",
            ),
            mean_answer_quality=(
                "answer_quality",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Category statistics
    # --------------------------------------------------

    category_stats = (
        successful
        .groupby("expected_label")
        .agg(
            observations=(
                "reward",
                "count",
            ),
            mean_reward=(
                "reward",
                "mean",
            ),
            mean_latency=(
                "latency_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Action/category matrix
    # --------------------------------------------------

    action_category = pd.crosstab(
        successful["expected_label"],
        successful["chosen_action"],
    )

    # --------------------------------------------------
    # Exploration statistics
    # --------------------------------------------------

    unique_actions = (
        successful["chosen_action"]
        .nunique()
    )

    total_actions = (
        len(successful)
    )

    exploration_rate = (
        1.0
        - (
            successful[
                "chosen_action"
            ].value_counts(
                normalize=True
            ).max()
        )
    )

    # --------------------------------------------------
    # Save detailed analysis
    # --------------------------------------------------

    analysis_columns = [
        "query_number",
        "query",
        "expected_label",
        "chosen_action",
        "selected_arm",
        "complexity_score",
        "reward",
        "cumulative_reward",
        "cumulative_average_reward",
        "answer_quality",
        "faithfulness",
        "retrieval_precision",
        "latency_seconds",
        "status",
    ]

    available_columns = [
        column
        for column in analysis_columns
        if column in successful.columns
    ]

    successful[
        available_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"\nTotal successful queries: "
        f"{total_actions}"
    )

    print(
        f"Unique actions selected: "
        f"{unique_actions}/3"
    )

    print(
        f"Average reward: "
        f"{successful['reward'].mean():.4f}"
    )

    print(
        f"Total cumulative reward: "
        f"{successful['reward'].sum():.4f}"
    )

    print(
        f"First-half reward: "
        f"{first_reward:.4f}"
    )

    print(
        f"Second-half reward: "
        f"{second_reward:.4f}"
    )

    print(
        f"Reward change: "
        f"{reward_change:+.4f}"
    )

    print(
        f"Exploration diversity score: "
        f"{exploration_rate:.4f}"
    )

    print(
        "\nAction statistics:"
    )

    print(
        action_stats.to_string(
            index=False
        )
    )

    print(
        "\nCategory statistics:"
    )

    print(
        category_stats.to_string(
            index=False
        )
    )

    print(
        "\nAction × category:"
    )

    print(
        action_category
    )

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 70
    )

    if unique_actions < 3:

        print(
            "\nThe policy did not explore all three actions "
            "during this 15-query run."
        )

    else:

        print(
            "\nThe policy explored all three available actions."
        )

    if reward_change > 0:

        print(
            "Average reward increased from the first half "
            "to the second half."
        )

    elif reward_change < 0:

        print(
            "Average reward decreased from the first half "
            "to the second half."
        )

        print(
            "This does not by itself indicate failed learning, "
            "because the query difficulty/order changed."
        )

    else:

        print(
            "Average reward remained approximately stable."
        )

    print(
        "\nThis 15-query run is a mechanism validation, "
        "not a generalization experiment."
    )

    print(
        "The proper learning evaluation will use the "
        "full 200-query dataset."
    )

    print(
        f"\nAnalysis saved to:\n{OUTPUT_FILE}"
    )

    print(
        "\nCheckpoint 4.6 complete."
    )


if __name__ == "__main__":
    main()