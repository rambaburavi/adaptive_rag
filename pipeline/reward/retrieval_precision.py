def score_retrieval_precision(evidence_chunks: list[dict]) -> float:
    """
    Estimate retrieval precision from the relevance scores
    already produced by retrieval/reranking.

    Returns:
        float between 0 and 1
    """

    if not evidence_chunks:
        return 0.0

    scores = []

    for chunk in evidence_chunks:
        score = chunk.get("score")

        if score is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        # Keep score within [0, 1].
        score = max(0.0, min(1.0, score))

        scores.append(score)

    if not scores:
        return 0.0

    # Average relevance of retrieved evidence.
    precision = sum(scores) / len(scores)

    return round(float(precision), 4)