from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = None


def _load_embedder():
    global _embedder

    if _embedder is None:
        print("Loading answer-quality embedding model...")
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedder


def _semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:

    if not text_a or not text_b:
        return 0.0

    embedder = _load_embedder()

    embeddings = embedder.encode(
        [text_a, text_b],
        normalize_embeddings=True,
    )

    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    normalized_similarity = (
        similarity + 1.0
    ) / 2.0

    return max(
        0.0,
        min(1.0, normalized_similarity),
    )


def score_answer_quality(
    answer: str,
    reference_text: str,
) -> float:
    """
    Score generated answer against evaluation reference text.

    The reference may contain multiple chunks separated by
    blank lines. The best semantic match is used.

    Returns a value between 0 and 1.
    """

    if not answer or not answer.strip():
        return 0.0

    if not reference_text or not reference_text.strip():
        return 0.0

    reference_chunks = [
        chunk.strip()
        for chunk in reference_text.split("\n\n")
        if chunk.strip()
    ]

    if not reference_chunks:
        return 0.0

    scores = []

    for reference_chunk in reference_chunks:

        score = _semantic_similarity(
            answer.strip(),
            reference_chunk,
        )

        scores.append(score)

    best_score = max(scores)

    return round(
        max(0.0, min(1.0, best_score)),
        4,
    )