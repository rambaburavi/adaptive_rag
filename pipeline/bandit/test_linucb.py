import numpy as np

from context import build_context
from linucb import LinUCB


def main():
    features = {
        "query_length": 32,
        "num_words": 7,
        "has_compare": False,
        "has_relationship": True,
        "has_how": True,
        "is_definition": False,
        "num_concepts": 2,
        "complexity_score": 0.40,
    }

    context = build_context(features)

    print("Context vector:")
    print(context)

    print("\nContext dimension:")
    print(len(context))

    bandit = LinUCB(
        n_arms=3,
        context_dim=8,
        alpha=1.0,
    )

    scores_before = bandit.predict(
        context
    )

    print("\nInitial UCB scores:")
    print(scores_before)

    selected_arm = bandit.select_arm(
        context
    )

    print("\nSelected arm:")
    print(selected_arm)

    reward = 0.75

    bandit.update(
        arm=selected_arm,
        context=context,
        reward=reward,
    )

    scores_after = bandit.predict(
        context
    )

    print("\nUpdated UCB scores:")
    print(scores_after)

    print("\nLinUCB test successful.")


if __name__ == "__main__":
    main()