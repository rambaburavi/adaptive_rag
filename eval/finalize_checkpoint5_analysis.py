from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import permutation_test


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    ROOT
    / "results"
    / "checkpoint5"
    / "checkpoint5_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "checkpoint5"
    / "final_analysis"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

SYSTEMS = [
    "LinUCB",
    "Manual Router",
    "Naive RAG",
]

BASELINES = [
    "Manual Router",
    "Naive RAG",
]

EXPECTED_QUERIES = 200

BOOTSTRAP_REPS = 20000
PERMUTATION_REPS = 20000
RANDOM_SEED = 42


# ============================================================
# GENERAL UTILITIES
# ============================================================

def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def holm_adjust(p_values):
    """
    Holm-Bonferroni correction for multiple hypothesis tests.
    """

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid = np.flatnonzero(np.isfinite(p_values))

    if len(valid) == 0:
        return adjusted

    ordered = valid[
        np.argsort(p_values[valid])
    ]

    number_of_tests = len(ordered)
    running_max = 0.0

    for rank, index in enumerate(ordered):

        candidate = min(
            1.0,
            (number_of_tests - rank)
            * p_values[index],
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = running_max

    return adjusted


def bootstrap_mean_ci(
    differences,
    seed=RANDOM_SEED,
    repetitions=BOOTSTRAP_REPS,
):
    """
    Percentile bootstrap confidence interval for
    the mean paired difference.

    Positive difference means LinUCB has the
    higher metric value.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)

    if n < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(repetitions)

    batch_size = 500

    for start in range(
        0,
        repetitions,
        batch_size,
    ):

        end = min(
            start + batch_size,
            repetitions,
        )

        indices = rng.integers(
            0,
            n,
            size=(end - start, n),
        )

        bootstrap_means[start:end] = (
            differences[indices].mean(axis=1)
        )

    lower = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper = np.percentile(
        bootstrap_means,
        97.5,
    )

    return float(lower), float(upper)


def paired_permutation_test(
    linucb_values,
    baseline_values,
    seed=RANDOM_SEED,
):
    """
    Paired permutation test of the mean difference.

    The null hypothesis is that the paired differences
    are exchangeable around zero under sign flipping.

    The statistic is the mean paired difference.
    """

    a = np.asarray(
        linucb_values,
        dtype=float,
    )

    b = np.asarray(
        baseline_values,
        dtype=float,
    )

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    differences = a - b

    n = len(differences)

    if n < 2:
        return {
            "n_pairs": n,
            "linucb_mean": np.nan,
            "baseline_mean": np.nan,
            "mean_difference": np.nan,
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "permutation_p": np.nan,
        }

    observed_difference = float(
        differences.mean()
    )

    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        seed=seed,
    )

    if np.allclose(differences, 0):

        p_value = 1.0

    else:

        def mean_difference_statistic(x, y):
            return np.mean(x - y)

        test_result = permutation_test(
            data=(
                a,
                b,
            ),
            statistic=mean_difference_statistic,
            permutation_type="samples",
            vectorized=False,
            n_resamples=PERMUTATION_REPS,
            alternative="two-sided",
            rng=np.random.default_rng(seed),
        )

        p_value = float(
            test_result.pvalue
        )

    return {
        "n_pairs": n,
        "linucb_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "mean_difference": observed_difference,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "permutation_p": p_value,
    }


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_results():

    if not RESULTS_FILE.exists():
        stop(
            f"Results file not found:\n"
            f"{RESULTS_FILE}"
        )

    df = pd.read_csv(
        RESULTS_FILE
    )

    required_columns = [
        "query_number",
        "category",
        "system",
        "action",
        "reward",
        "answer_quality",
        "latency_seconds",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        stop(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    if len(df) != 600:
        stop(
            f"Expected 600 rows; found {len(df)}."
        )

    df["system"] = (
        df["system"]
        .astype(str)
        .str.strip()
    )

    df["action"] = (
        df["action"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["query_number"] = pd.to_numeric(
        df["query_number"],
        errors="coerce",
    )

    for column in [
        "reward",
        "answer_quality",
        "latency_seconds",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[
        [
            "query_number",
            "reward",
            "answer_quality",
            "latency_seconds",
        ]
    ].isna().any().any():

        stop(
            "Missing or invalid values found in "
            "required analysis columns."
        )

    if df.duplicated(
        subset=["system", "query_number"]
    ).any():

        stop(
            "Duplicate system/query pairs detected."
        )

    counts = df["system"].value_counts()

    for system in SYSTEMS:

        if counts.get(system, 0) != EXPECTED_QUERIES:
            stop(
                f"{system}: expected "
                f"{EXPECTED_QUERIES} rows; "
                f"found {counts.get(system, 0)}."
            )

    query_sets = {
        system: set(
            df.loc[
                df["system"] == system,
                "query_number",
            ]
        )
        for system in SYSTEMS
    }

    reference_set = query_sets["LinUCB"]

    for system in BASELINES:

        if query_sets[system] != reference_set:
            stop(
                f"Query coverage mismatch for {system}."
            )

    print("=" * 75)
    print("FINAL CP5 STATISTICAL ANALYSIS")
    print("=" * 75)

    print(f"\nRows validated: {len(df)}")

    for system in SYSTEMS:
        print(
            f"{system}: "
            f"{counts[system]} evaluations"
        )

    print(
        "\nQuery coverage, uniqueness, and required "
        "metric completeness passed."
    )

    return df


# ============================================================
# A. STATISTICAL AUDIT
# ============================================================

def run_primary_statistical_analysis(df):

    print("\n" + "=" * 75)
    print("A. COMMON-METRIC STATISTICAL ANALYSIS")
    print("=" * 75)

    print(
        "\nPrimary outcome: answer_quality."
    )

    print(
        "All comparisons use the same query IDs "
        "and the same answer-quality metric."
    )

    print(
        "Test: paired permutation test of the mean "
        "difference; two-sided."
    )

    print(
        "Confidence interval: 95% paired bootstrap "
        "interval for the mean difference."
    )

    print(
        "Multiplicity correction: Holm correction "
        "across the two primary baseline comparisons."
    )

    rows = []

    for baseline in BASELINES:

        pivot = df.pivot(
            index="query_number",
            columns="system",
            values="answer_quality",
        )

        paired = pivot[
            [
                "LinUCB",
                baseline,
            ]
        ].dropna()

        result = paired_permutation_test(
            paired["LinUCB"].to_numpy(),
            paired[baseline].to_numpy(),
            seed=RANDOM_SEED,
        )

        result["metric"] = "answer_quality"

        result["comparison"] = (
            f"LinUCB vs {baseline}"
        )

        rows.append(result)

    results = pd.DataFrame(rows)

    results["holm_adjusted_p"] = holm_adjust(
        results["permutation_p"].to_numpy()
    )

    results["significant_at_0.05"] = (
        results["holm_adjusted_p"] < 0.05
    )

    results = results[
        [
            "metric",
            "comparison",
            "n_pairs",
            "linucb_mean",
            "baseline_mean",
            "mean_difference",
            "ci_95_low",
            "ci_95_high",
            "permutation_p",
            "holm_adjusted_p",
            "significant_at_0.05",
        ]
    ]

    output_file = (
        OUTPUT_DIR
        / "common_metric_statistical_tests.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nPrimary statistical results:"
    )

    print(
        results.round(6).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output_file}"
    )

    # Secondary descriptive metrics.
    # Do not interpret reward as an action-independent outcome.

    descriptive = (
        df.groupby("system")[
            [
                "reward",
                "answer_quality",
                "latency_seconds",
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
            ]
        )
    )

    descriptive_file = (
        OUTPUT_DIR
        / "common_metric_descriptive_summary.csv"
    )

    descriptive.to_csv(
        descriptive_file
    )

    print(
        f"Saved: {descriptive_file}"
    )

    return results


# ============================================================
# FIGURE 1. OVERALL PERFORMANCE
# ============================================================

def plot_overall_performance(df):

    print(
        "\nGenerating Figure 1: Overall performance..."
    )

    summary = (
        df.groupby("system")[
            [
                "reward",
                "answer_quality",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    order = SYSTEMS

    reward_means = [
        summary.loc[system, ("reward", "mean")]
        for system in order
    ]

    reward_stds = [
        summary.loc[system, ("reward", "std")]
        for system in order
    ]

    quality_means = [
        summary.loc[
            system,
            ("answer_quality", "mean"),
        ]
        for system in order
    ]

    quality_stds = [
        summary.loc[
            system,
            ("answer_quality", "std"),
        ]
        for system in order
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    x = np.arange(len(order))

    axes[0].bar(
        x,
        reward_means,
        yerr=reward_stds,
        capsize=5,
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        order,
        rotation=15,
        ha="right",
    )

    axes[0].set_ylabel(
        "Mean implementation-defined reward"
    )

    axes[0].set_title(
        "Overall reward"
    )

    axes[0].set_ylim(bottom=0)

    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    axes[1].bar(
        x,
        quality_means,
        yerr=quality_stds,
        capsize=5,
    )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        order,
        rotation=15,
        ha="right",
    )

    axes[1].set_ylabel(
        "Reference-based answer quality"
    )

    axes[1].set_title(
        "Overall answer quality"
    )

    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    fig.suptitle(
        "Overall comparison across 200 evaluation queries"
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "figure1_overall_performance.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# FIGURE 2. CATEGORY COMPARISON
# ============================================================

def plot_category_comparison(df):

    print(
        "\nGenerating Figure 2: Category comparison..."
    )

    category_order = [
        "simple",
        "multi-hop",
        "comparison",
    ]

    summary = (
        df.groupby(
            [
                "category",
                "system",
            ]
        )["reward"]
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    x = np.arange(
        len(category_order)
    )

    width = 0.25

    for index, system in enumerate(SYSTEMS):

        system_summary = (
            summary[
                summary["system"] == system
            ]
            .set_index("category")
            .reindex(category_order)
        )

        means = system_summary[
            "mean"
        ].to_numpy()

        stds = system_summary[
            "std"
        ].to_numpy()

        positions = (
            x
            + (index - 1) * width
        )

        ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=system,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        category_order
    )

    ax.set_ylabel(
        "Mean implementation-defined reward"
    )

    ax.set_xlabel(
        "Query category"
    )

    ax.set_title(
        "Reward by query category"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "figure2_category_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# FIGURE 3. LINUCB ACTION DISTRIBUTION
# ============================================================

def plot_action_distribution(df):

    print(
        "\nGenerating Figure 3: LinUCB action distribution..."
    )

    linucb = df[
        df["system"] == "LinUCB"
    ].copy()

    action_labels = {
        "answer_directly": "Answer directly",
        "retrieve": "Retrieve",
        "decompose": "Decompose",
    }

    counts = (
        linucb["action"]
        .value_counts()
        .reindex(
            action_labels.keys(),
            fill_value=0,
        )
    )

    percentages = (
        counts / len(linucb)
    ) * 100

    labels = [
        action_labels[action]
        for action in counts.index
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    bars = ax.bar(
        labels,
        counts.to_numpy(),
    )

    for bar, count, percentage in zip(
        bars,
        counts.to_numpy(),
        percentages.to_numpy(),
    ):

        ax.annotate(
            (
                f"{count} queries\n"
                f"({percentage:.1f}%)"
            ),
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel(
        "Number of queries"
    )

    ax.set_xlabel(
        "Selected action"
    )

    ax.set_title(
        "LinUCB action distribution across 200 queries"
    )

    ax.set_ylim(
        bottom=0,
        top=max(counts.max() * 1.25, 1),
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "figure3_action_distribution.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    action_table = pd.DataFrame({
        "action": labels,
        "count": counts.to_numpy(),
        "percentage": percentages.to_numpy(),
    })

    table_file = (
        OUTPUT_DIR
        / "linucb_action_distribution.csv"
    )

    action_table.to_csv(
        table_file,
        index=False,
    )

    print(
        action_table.to_string(
            index=False
        )
    )

    print(f"Saved: {output_file}")
    print(f"Saved: {table_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    run_primary_statistical_analysis(df)

    plot_overall_performance(df)

    plot_category_comparison(df)

    plot_action_distribution(df)

    print("\n" + "=" * 75)
    print("FINAL ANALYSIS COMPLETE")
    print("=" * 75)

    print(
        "\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nFigures:"
        "\n1. figure1_overall_performance.png"
        "\n2. figure2_category_comparison.png"
        "\n3. figure3_action_distribution.png"
    )

    print(
        "\nStatistical report:"
        "\ncommon_metric_statistical_tests.csv"
    )

    print(
        "\nImportant: the common answer-quality metric "
        "supports a more comparable outcome analysis. "
        "The implementation-defined reward remains "
        "secondary because its formula depends on action."
    )


if __name__ == "__main__":
    main()