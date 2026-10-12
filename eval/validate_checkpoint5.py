from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "checkpoint5" / "checkpoint5_results.csv"
OUTPUT = ROOT / "results" / "checkpoint5"

EXPECTED_ROWS = 600
EXPECTED_QUERIES_PER_SYSTEM = 200

SYSTEMS = [
    "LinUCB",
    "Manual Router",
    "Naive RAG",
]

METRICS = [
    "reward",
    "answer_quality",
    "faithfulness",
    "retrieval_relevance",
    "latency_seconds",
    "latency_score",
]

BOUNDS_01_METRICS = [
    "reward",
    "answer_quality",
    "faithfulness",
    "retrieval_relevance",
    "latency_score",
]

EXPECTED_CATEGORIES = {
    "simple": 80,
    "multi-hop": 70,
    "comparison": 50,
}

BOOTSTRAP_REPS = 10000
RANDOM_SEED = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def find_column(
    df,
    candidates,
    description,
    required=True,
):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        stop(
            f"Cannot identify {description} column.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected one of: {candidates}"
        )

    return None


def bootstrap_mean_ci(
    values,
    rng,
    reps=BOOTSTRAP_REPS,
):
    """
    Percentile bootstrap confidence interval for a mean.

    Returns:
        mean, lower_95_ci, upper_95_ci
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        return (
            float(values[0]),
            np.nan,
            np.nan,
        )

    bootstrap_means = np.empty(reps)

    batch_size = 500

    for start in range(0, reps, batch_size):
        stop_index = min(start + batch_size, reps)
        current_batch = stop_index - start

        indices = rng.integers(
            0,
            len(values),
            size=(current_batch, len(values)),
        )

        bootstrap_means[start:stop_index] = (
            values[indices].mean(axis=1)
        )

    return (
        float(values.mean()),
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def paired_test(
    system_values,
    baseline_values,
    seed=RANDOM_SEED,
):
    """
    Compare paired observations.

    Difference = system_values - baseline_values.

    Positive differences favor LinUCB for metrics where
    higher is better.

    Negative differences favor LinUCB for latency_seconds.
    """

    a = np.asarray(system_values, dtype=float)
    b = np.asarray(baseline_values, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    differences = a - b
    n = len(differences)

    result = {
        "n_pairs": n,
        "linucb_mean": float(np.mean(a)) if n else np.nan,
        "baseline_mean": float(np.mean(b)) if n else np.nan,
        "mean_difference": np.nan,
        "ci_95_low": np.nan,
        "ci_95_high": np.nan,
        "wilcoxon_p": np.nan,
    }

    if n < 2:
        return result

    rng = np.random.default_rng(seed)

    mean_diff, ci_low, ci_high = bootstrap_mean_ci(
        differences,
        rng,
    )

    result["mean_difference"] = mean_diff
    result["ci_95_low"] = ci_low
    result["ci_95_high"] = ci_high

    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        try:
            p_value = float(
                wilcoxon(
                    differences,
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            p_value = np.nan

    result["wilcoxon_p"] = p_value

    return result


def holm_adjust(p_values):
    """
    Holm-Bonferroni multiple-testing correction.

    NaN p-values remain NaN.
    """

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.flatnonzero(
        np.isfinite(p_values)
    )

    if len(valid_indices) == 0:
        return adjusted

    order = valid_indices[
        np.argsort(p_values[valid_indices])
    ]

    m = len(order)
    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (m - rank) * p_values[index],
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = running_max

    return adjusted


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    if not INPUT.exists():
        stop(
            f"Results file not found:\n{INPUT}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    print("=" * 75)
    print("CHECKPOINT 5: DATA VALIDATION AND STATISTICAL ANALYSIS")
    print("=" * 75)

    print(f"\nInput file: {INPUT}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    problems = []

    # --------------------------------------------------------
    # 1. IDENTIFY COLUMNS
    # --------------------------------------------------------

    system_col = find_column(
        df,
        ["system", "model", "method", "approach"],
        "system",
    )

    query_id_col = find_column(
        df,
        [
            "query_number",
            "query_id",
            "question_id",
            "query_index",
        ],
        "query ID",
        required=False,
    )

    query_text_col = find_column(
        df,
        ["query", "question", "query_text"],
        "query text",
        required=False,
    )

    category_col = find_column(
        df,
        ["category", "query_category", "type"],
        "query category",
        required=False,
    )

    action_col = find_column(
        df,
        ["action", "chosen_action", "selected_action"],
        "action",
    )

    if query_id_col is not None:
        query_key = query_id_col
    elif query_text_col is not None:
        query_key = query_text_col
    else:
        stop(
            "No query ID or query text column found. "
            "A stable query key is required."
        )

    # --------------------------------------------------------
    # 2. NORMALIZE SYSTEM AND ACTION LABELS
    # --------------------------------------------------------

    df[system_col] = (
        df[system_col]
        .astype(str)
        .str.strip()
    )

    system_aliases = {
        "linucb": "LinUCB",
        "manual router": "Manual Router",
        "manual_router": "Manual Router",
        "naive rag": "Naive RAG",
        "naive_rag": "Naive RAG",
    }

    df[system_col] = df[system_col].map(
        lambda value: system_aliases.get(
            value.lower(),
            value,
        )
    )

    df[action_col] = (
        df[action_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # 3. ROW COUNTS
    # --------------------------------------------------------

    print("\n1. ROW COUNT AND SYSTEM DISTRIBUTION")
    print("-" * 75)

    print(
        df[system_col]
        .value_counts(dropna=False)
        .to_string()
    )

    if len(df) != EXPECTED_ROWS:
        problems.append(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    actual_systems = set(df[system_col].dropna())
    expected_systems = set(SYSTEMS)

    if actual_systems != expected_systems:
        problems.append(
            f"System mismatch. Expected {expected_systems}; "
            f"found {actual_systems}."
        )

    system_counts = df[system_col].value_counts()

    for system in SYSTEMS:
        count = int(system_counts.get(system, 0))

        if count != EXPECTED_QUERIES_PER_SYSTEM:
            problems.append(
                f"{system}: expected "
                f"{EXPECTED_QUERIES_PER_SYSTEM} rows, "
                f"found {count}."
            )

    # --------------------------------------------------------
    # 4. DUPLICATE EVALUATIONS
    # --------------------------------------------------------

    print("\n2. DUPLICATE EVALUATION CHECK")
    print("-" * 75)

    duplicate_mask = df.duplicated(
        subset=[system_col, query_key],
        keep=False,
    )

    duplicate_rows = df.loc[duplicate_mask]

    print(
        "Duplicate system/query rows:",
        len(duplicate_rows),
    )

    if not duplicate_rows.empty:
        print(
            duplicate_rows[
                [system_col, query_key]
            ].sort_values(
                [system_col, query_key]
            ).to_string(index=False)
        )

        problems.append(
            "Duplicate system/query evaluations found."
        )

    if df[query_key].isna().any():
        problems.append(
            "Missing query IDs or query text found."
        )

    query_sets = {
        system: set(
            df.loc[
                df[system_col] == system,
                query_key,
            ].dropna()
        )
        for system in SYSTEMS
    }

    print(
        "\nUnique query keys overall:",
        df[query_key].nunique(),
    )

    for system in SYSTEMS:
        print(
            f"{system}: "
            f"{len(query_sets[system])} unique queries"
        )

    reference_query_set = query_sets[SYSTEMS[0]]

    for system in SYSTEMS[1:]:
        if query_sets[system] != reference_query_set:
            problems.append(
                f"{system} does not contain exactly the "
                "same query set as LinUCB."
            )

    # --------------------------------------------------------
    # 5. METRIC COMPLETENESS AND VALIDITY
    # --------------------------------------------------------

    print("\n3. METRIC COMPLETENESS AND VALIDITY")
    print("-" * 75)

    metric_columns = {}

    for metric in METRICS:

        col = find_column(
            df,
            [metric],
            metric,
            required=False,
        )

        if col is None:
            print(f"{metric}: COLUMN MISSING")
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        metric_columns[metric] = col

        missing = int(df[col].isna().sum())

        infinite = int(
            np.isinf(
                df[col].dropna().to_numpy(dtype=float)
            ).sum()
        )

        print(
            f"{metric}: "
            f"missing/non-numeric={missing}, "
            f"infinite={infinite}"
        )

        if infinite:
            problems.append(
                f"{metric} contains {infinite} infinite values."
            )

        if missing:

            if metric in [
                "faithfulness",
                "retrieval_relevance",
            ]:

                direct_mask = (
                    df[action_col] == "answer_directly"
                )

                missing_on_direct = int(
                    df.loc[
                        direct_mask,
                        col,
                    ].isna().sum()
                )

                missing_on_other_actions = int(
                    df.loc[
                        ~direct_mask,
                        col,
                    ].isna().sum()
                )

                print(
                    "  Missing on answer_directly:",
                    missing_on_direct,
                )

                print(
                    "  Missing on other actions:",
                    missing_on_other_actions,
                )

                if missing_on_other_actions:
                    problems.append(
                        f"{metric} has missing values for "
                        "non-direct actions. Investigate."
                    )

                if missing_on_direct != missing:
                    problems.append(
                        f"{metric} has missing values outside "
                        "the expected direct-answer pattern."
                    )

            else:
                problems.append(
                    f"{metric} has {missing} missing values."
                )

        if metric in BOUNDS_01_METRICS:

            outside = (
                (df[col] < 0)
                | (df[col] > 1)
            ) & df[col].notna()

            if outside.any():
                problems.append(
                    f"{metric} contains values outside [0, 1]."
                )

        if metric == "latency_seconds":

            if (df[col] < 0).any():
                problems.append(
                    "Negative latency values found."
                )

    required_metrics = {
        "reward",
        "answer_quality",
        "latency_seconds",
        "latency_score",
    }

    missing_required = (
        required_metrics - set(metric_columns)
    )

    if missing_required:
        stop(
            f"Missing required metric columns: "
            f"{missing_required}"
        )

    # --------------------------------------------------------
    # 6. RECOMPUTE OVERALL METRICS
    # --------------------------------------------------------

    print("\n4. RECOMPUTED OVERALL METRICS")
    print("-" * 75)

    summary_rows = []

    for system in SYSTEMS:

        system_df = df[
            df[system_col] == system
        ]

        for metric, col in metric_columns.items():

            values = system_df[col].dropna()

            summary_rows.append({
                "system": system,
                "metric": metric,
                "count": len(values),
                "mean": (
                    float(values.mean())
                    if len(values)
                    else np.nan
                ),
                "std": (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_pivot = summary_df.pivot(
        index="system",
        columns="metric",
        values="mean",
    )

    print(
        summary_pivot.round(4).to_string()
    )

    summary_df.to_csv(
        OUTPUT / "validation_recomputed_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    if category_col is not None:

        print("\n5. CATEGORY COUNTS")
        print("-" * 75)

        df["_normalized_category"] = (
            df[category_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        category_counts = pd.crosstab(
            df["_normalized_category"],
            df[system_col],
        )

        print(
            category_counts.to_string()
        )

        for category, expected_count in (
            EXPECTED_CATEGORIES.items()
        ):

            for system in SYSTEMS:

                actual_count = int(
                    (
                        (df["_normalized_category"] == category)
                        & (df[system_col] == system)
                    ).sum()
                )

                print(
                    f"{system} / {category}: "
                    f"{actual_count}"
                )

                if actual_count != expected_count:
                    problems.append(
                        f"{system}/{category}: expected "
                        f"{expected_count}, found {actual_count}."
                    )

        category_summary = (
            df.groupby(
                [system_col, "_normalized_category"]
            )[list(metric_columns.values())]
            .agg(["count", "mean", "std"])
        )

        category_summary.to_csv(
            OUTPUT / "validation_category_summary.csv"
        )

    # --------------------------------------------------------
    # 8. VALIDATION VERDICT
    # --------------------------------------------------------

    print("\n6. VALIDATION VERDICT")
    print("-" * 75)

    if problems:

        print("VALIDATION FAILED / NEEDS REVIEW")

        for problem in problems:
            print(f"- {problem}")

        stop(
            "Resolve the listed issues before interpreting "
            "statistical results."
        )

    print(
        "PASS: row counts, query coverage, category counts, "
        "and metric validation."
    )

    print(
        "\nNote: faithfulness and retrieval_relevance are "
        "undefined for direct-answer rows in this dataset. "
        "They are not imputed with zero."
    )

    # --------------------------------------------------------
    # 9. PAIRED STATISTICAL TESTS
    # --------------------------------------------------------

    print("\n7. PAIRED STATISTICAL TESTS")
    print("-" * 75)

    print(
        "Difference = LinUCB minus baseline."
    )

    print(
        "For reward, answer_quality, and latency_score, "
        "positive differences favor LinUCB."
    )

    print(
        "For latency_seconds, negative differences favor LinUCB."
    )

    test_rows = []

    # Evaluate all available metrics.
    for metric, col in metric_columns.items():

        pivot = df.pivot(
            index=query_key,
            columns=system_col,
            values=col,
        )

        for baseline in [
            "Manual Router",
            "Naive RAG",
        ]:

            pair = pivot[
                ["LinUCB", baseline]
            ].dropna()

            result = paired_test(
                pair["LinUCB"].to_numpy(),
                pair[baseline].to_numpy(),
                seed=RANDOM_SEED,
            )

            result.update({
                "metric": metric,
                "comparison": f"LinUCB vs {baseline}",
            })

            test_rows.append(result)

    tests = pd.DataFrame(test_rows)

    tests["holm_adjusted_p"] = holm_adjust(
        tests["wilcoxon_p"].to_numpy()
    )

    tests["significant_at_0.05"] = (
        tests["holm_adjusted_p"] < 0.05
    )

    column_order = [
        "metric",
        "comparison",
        "n_pairs",
        "linucb_mean",
        "baseline_mean",
        "mean_difference",
        "ci_95_low",
        "ci_95_high",
        "wilcoxon_p",
        "holm_adjusted_p",
        "significant_at_0.05",
    ]

    tests = tests[column_order]

    print(
        tests.round(6).to_string(index=False)
    )

    tests.to_csv(
        OUTPUT / "validation_paired_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. CATEGORY-SPECIFIC PAIRED TESTS
    # --------------------------------------------------------

    if category_col is not None:

        print("\n8. CATEGORY-SPECIFIC REWARD COMPARISONS")
        print("-" * 75)

        category_test_rows = []

        reward_col = metric_columns["reward"]

        for category in EXPECTED_CATEGORIES:

            category_df = df[
                df["_normalized_category"] == category
            ]

            pivot = category_df.pivot(
                index=query_key,
                columns=system_col,
                values=reward_col,
            )

            for baseline in [
                "Manual Router",
                "Naive RAG",
            ]:

                pair = pivot[
                    ["LinUCB", baseline]
                ].dropna()

                result = paired_test(
                    pair["LinUCB"].to_numpy(),
                    pair[baseline].to_numpy(),
                    seed=RANDOM_SEED,
                )

                result.update({
                    "category": category,
                    "comparison": f"LinUCB vs {baseline}",
                })

                category_test_rows.append(result)

        category_tests = pd.DataFrame(
            category_test_rows
        )

        category_tests["holm_adjusted_p"] = holm_adjust(
            category_tests["wilcoxon_p"].to_numpy()
        )

        category_tests["significant_at_0.05"] = (
            category_tests["holm_adjusted_p"] < 0.05
        )

        print(
            category_tests.round(6).to_string(index=False)
        )

        category_tests.to_csv(
            OUTPUT / "validation_category_statistics.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 11. OUTPUT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VALIDATION AND STATISTICAL ANALYSIS COMPLETE")
    print("=" * 75)

    print("\nGenerated files:")

    print(
        OUTPUT / "validation_recomputed_summary.csv"
    )

    print(
        OUTPUT / "validation_category_summary.csv"
    )

    print(
        OUTPUT / "validation_paired_statistics.csv"
    )

    if category_col is not None:
        print(
            OUTPUT / "validation_category_statistics.csv"
        )

    print(
        "\nResearch cautions:"
        "\n1. Statistical significance does not establish "
        "generalization to unseen queries."
        "\n2. The current reward formula uses different "
        "components for different actions."
        "\n3. Retrieval-specific metrics are evaluated only "
        "on valid paired observations."
        "\n4. Report sample sizes, confidence intervals, "
        "effect sizes, and limitations."
    )


if __name__ == "__main__":
    main()