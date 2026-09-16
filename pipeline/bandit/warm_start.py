import os
import sys

import pandas as pd

PIPELINE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from feature_extractor import extract_features
from bandit.context import build_context
from bandit.linucb import LinUCB


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


def warm_start_from_results(
    results_df: pd.DataFrame,
    alpha: float = 1.0,
) -> LinUCB:
    bandit = LinUCB(
        n_arms=3,
        context_dim=8,
        alpha=alpha,
    )

    for _, row in results_df.iterrows():

        action = str(
            row["chosen_action"]
        )

        if action not in ACTION_TO_ARM:
            continue

        reward = float(
            row["reward"]
        )

        query = str(
            row["query"]
        )

        features = extract_features(
            query
        )

        context = build_context(
            features
        )

        arm = ACTION_TO_ARM[action]

        bandit.update(
            arm=arm,
            context=context,
            reward=reward,
        )

    return bandit