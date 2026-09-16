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
    "checkpoint4_5_analysis.csv",
)


def main():

    print("=" * 70)
    print("CHECKPOINT 4.5 - LINUCB LEARNING TRAJECTORY")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nLoaded results: {len(df)}"
    )

    successful = df[
        df["status"] == "success"
    ].copy()

    if successful.empty:
        raise ValueError(
            "No successful queries found."
        )

    # --------------------------------------------------
    # Add query position
    # --------------------------------------------------

    successful["query_number"] = (
        range(1, len(successful) + 1)
    )

    # --------------------------------------------------
    # Cumulative average reward
    # --------------------------------------------------

    successful["cumulative_average_reward"] = (
        successful["reward"].cumsum()
        /
        successful["query_number"]
    )

    # --------------------------------------------------
    # Rolling reward
    # --------------------------------------------------

    successful["rolling_reward_3"] = (
        successful["reward"]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------
    # First half vs second half
    # --------------------------------------------------

    midpoint = len(successful) // 2

    first_half = successful.iloc[
        :midpoint
    ]

    second_half = successful.iloc[
        midpoint:
    ]

    first_half_reward = (
        first_half["reward"].mean()
    )

    second_half_reward = (
        second_half["reward"].mean()
    )

    reward_change = (
        second_half_reward
        - first_half_reward
    )

    # --------------------------------------------------
    # Action transitions
    # --------------------------------------------------

    actions = (
        successful["chosen_action"]
        .tolist()
    )

    transitions = 0

    for i in range(1, len(actions)):

        if actions[i] != actions[i - 1]:
            transitions += 1

    # --------------------------------------------------
    # Action statistics
    # --------------------------------------------------

    action_stats = (
        successful
        .groupby("chosen_action")
        .agg(
            count=("reward", "count"),
            average_reward=("reward", "mean"),
            average_latency=("latency_seconds", "mean"),
            average_answer_quality=(
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
            count=("reward", "count"),
            average_reward=("reward", "mean"),
            average_latency=("latency_seconds", "mean"),
        )
        .reset_index()
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
        "rolling_reward_3",
        "latency_seconds",
        "answer_quality",
        "faithfulness",
        "retrieval_precision",
        "status",
    ]

    analysis_df = successful[
        [
            column
            for column in analysis_columns
            if column in successful.columns
        ]
    ].copy()

    analysis_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Print trajectory
    # --------------------------------------------------

    print(
        "\nLEARNING TRAJECTORY"
    )

    print("-" * 70)

    for _, row in successful.iterrows():

        print(
            f"Q{int(row['query_number']):02d} | "
            f"{row['chosen_action']:<17} | "
            f"Reward: {row['reward']:.4f} | "
            f"CumAvg: "
            f"{row['cumulative_average_reward']:.4f}"
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CP4.5 SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal successful queries: "
        f"{len(successful)}"
    )

    print(
        f"Average reward: "
        f"{successful['reward'].mean():.4f}"
    )

    print(
        f"First half average reward: "
        f"{first_half_reward:.4f}"
    )

    print(
        f"Second half average reward: "
        f"{second_half_reward:.4f}"
    )

    print(
        f"Reward change: "
        f"{reward_change:+.4f}"
    )

    print(
        f"\nAction transitions: "
        f"{transitions}"
    )

    print(
        "\nAction distribution:"
    )

    print(
        successful[
            "chosen_action"
        ].value_counts()
    )

    print(
        "\nReward by action:"
    )

    for _, row in action_stats.iterrows():

        print(
            f"{row['chosen_action']:<17} "
            f"Count={int(row['count']):2d} | "
            f"Reward={row['average_reward']:.4f} | "
            f"Latency={row['average_latency']:.2f}s | "
            f"AQ={row['average_answer_quality']:.4f}"
        )

    print(
        "\nReward by evaluation category:"
    )

    for _, row in category_stats.iterrows():

        print(
            f"{row['expected_label']:<12} "
            f"Count={int(row['count']):2d} | "
            f"Reward={row['average_reward']:.4f} | "
            f"Latency={row['average_latency']:.2f}s"
        )

    print(
        "\nSaved detailed analysis:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nCP4.5 mechanism analysis complete."
    )


if __name__ == "__main__":
    main()