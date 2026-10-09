import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoint5"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "checkpoint5_results.csv"
)

PLOT_DIR = RESULTS_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(
    RESULTS_FILE
)


# ============================================================
# 1. LINUCB CUMULATIVE / ROLLING REWARD
# ============================================================

linucb = (
    df[df["system"] == "LinUCB"]
    .sort_values("query_number")
    .copy()
)

linucb["cumulative_reward"] = (
    linucb["reward"]
    .fillna(0)
    .cumsum()
)

linucb["rolling_reward"] = (
    linucb["reward"]
    .rolling(
        window=20,
        min_periods=1,
    )
    .mean()
)


plt.figure(figsize=(10, 6))

plt.plot(
    linucb["query_number"],
    linucb["cumulative_reward"],
    label="Cumulative Reward",
)

plt.xlabel("Query Number")
plt.ylabel("Cumulative Reward")
plt.title(
    "LinUCB Cumulative Reward Across 200 Queries"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "linucb_cumulative_reward.png",
    dpi=300,
)

plt.close()


plt.figure(figsize=(10, 6))

plt.plot(
    linucb["query_number"],
    linucb["rolling_reward"],
)

plt.xlabel("Query Number")
plt.ylabel("Rolling Average Reward")
plt.title(
    "LinUCB Rolling Average Reward (Window = 20)"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "linucb_rolling_reward.png",
    dpi=300,
)

plt.close()


# ============================================================
# 2. ACTION DISTRIBUTION
# ============================================================

action_counts = (
    pd.crosstab(
        df["system"],
        df["action"],
    )
)

action_counts.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.xlabel("System")
plt.ylabel("Number of Queries")
plt.title(
    "Action Distribution by System"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "action_distribution.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. OVERALL REWARD
# ============================================================

overall_reward = (
    df.groupby("system")["reward"]
    .mean()
)

overall_reward.plot(
    kind="bar",
    figsize=(8, 6),
)

plt.xlabel("System")
plt.ylabel("Average Reward")
plt.title(
    "Average Reward Across Systems"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "overall_reward_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. REWARD BY CATEGORY
# ============================================================

category_reward = (
    df.groupby(
        ["category", "system"]
    )["reward"]
    .mean()
    .unstack()
)

category_reward.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.xlabel("Query Category")
plt.ylabel("Average Reward")
plt.title(
    "Average Reward by Query Category"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "category_reward_comparison.png",
    dpi=300,
)

plt.close()


print(
    "CP5 plots generated successfully."
)

print(
    f"Saved to: {PLOT_DIR}"
)