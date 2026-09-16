import numpy as np

FEATURE_ORDER = [
    "query_length",
    "num_words",
    "has_compare",
    "has_relationship",
    "has_how",
    "is_definition",
    "num_concepts",
    "complexity_score",
]

FEATURE_MIN = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
], dtype=float)

FEATURE_MAX = np.array([
    300.0,
    60.0,
    1.0,
    1.0,
    1.0,
    1.0,
    10.0,
    1.0,
], dtype=float)


def build_context(features: dict) -> np.ndarray:
    values = np.array(
        [
            float(features.get(name, 0.0))
            for name in FEATURE_ORDER
        ],
        dtype=float,
    )

    normalized = (
        values - FEATURE_MIN
    ) / (
        FEATURE_MAX - FEATURE_MIN
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized