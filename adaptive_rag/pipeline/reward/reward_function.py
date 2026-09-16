from .faithfulness import score_faithfulness
from .retrieval_precision import score_retrieval_precision
from .latency import latency_score
from .answer_quality import score_answer_quality


FAITHFULNESS_WEIGHT = 0.4
ANSWER_QUALITY_WEIGHT = 0.3
RETRIEVAL_WEIGHT = 0.2
LATENCY_WEIGHT = 0.1


def compute_reward(
    query,
    answer,
    evidence_chunks,
    latency_seconds,
    action_name=None,
    reference_text=None,
):
    """
    Compute the final reward for an action.

    Retrieval-based actions:
        answer quality
        faithfulness
        retrieval quality
        latency

    Direct-answer actions:
        answer quality
        latency

    reference_text is used only during evaluation.
    It is NOT provided to the answering action.
    """

    latency = latency_score(latency_seconds)

    # ---------------------------------------------------------
    # Answer quality
    # ---------------------------------------------------------

    if reference_text:
        answer_quality = score_answer_quality(
            answer=answer,
            reference_text=reference_text,
        )
    else:
        answer_quality = 0.0

    # ---------------------------------------------------------
    # Direct answer
    # ---------------------------------------------------------

    if action_name == "answer_directly":

        final_reward = (
            0.8 * answer_quality
            + 0.2 * latency
        )

        return {
            "query": query,
            "action": action_name,
            "answer_quality": round(answer_quality, 4),
            "faithfulness": None,
            "retrieval_precision": None,
            "latency_score": round(latency, 4),
            "latency_seconds": round(latency_seconds, 4),
            "reward": round(
                max(0.0, min(1.0, final_reward)),
                4,
            ),
        }

    # ---------------------------------------------------------
    # Retrieval-based actions
    # ---------------------------------------------------------

    faithfulness = score_faithfulness(
        answer,
        evidence_chunks,
    )

    retrieval_precision = score_retrieval_precision(
        evidence_chunks
    )

    final_reward = (
        FAITHFULNESS_WEIGHT * faithfulness
        + ANSWER_QUALITY_WEIGHT * answer_quality
        + RETRIEVAL_WEIGHT * retrieval_precision
        + LATENCY_WEIGHT * latency
    )

    return {
        "query": query,
        "action": action_name,
        "answer_quality": round(answer_quality, 4),
        "faithfulness": round(faithfulness, 4),
        "retrieval_precision": round(
            retrieval_precision,
            4,
        ),
        "latency_score": round(latency, 4),
        "latency_seconds": round(
            latency_seconds,
            4,
        ),
        "reward": round(
            max(0.0, min(1.0, final_reward)),
            4,
        ),
    }