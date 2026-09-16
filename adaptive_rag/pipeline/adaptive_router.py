from feature_extractor import extract_features

def adaptive_route(query: str) -> tuple[str, dict]:
    """
    Transparent, rule-based controller for Checkpoint 2.
    Returns (action_name, features) so callers can log the decision basis.
    This will later be replaced by the learned LinUCB policy in Checkpoint 4 —
    keep this function's signature stable so swapping it in is a drop-in change.
    """
    features = extract_features(query)

    # very simple, single-concept, definition-style -> skip retrieval entirely
    if features["is_definition"] and features["num_concepts"] == 1 and features["complexity_score"] < 0.25:
        return "answer_directly", features

    # explicit comparison or multi-concept relationship queries -> decompose
    if features["has_compare"] or (features["has_relationship"] and features["num_concepts"] > 1):
        return "decompose", features

    # everything else with moderate-to-high complexity -> retrieve
    if features["complexity_score"] >= 0.3:
        return "retrieve", features

    # default fallback
    return "retrieve", features