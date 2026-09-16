from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import re


# =========================================================
# Models
# =========================================================

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


_tokenizer = None
_nli_model = None
_embedder = None


# =========================================================
# Model loading
# =========================================================

def _load_nli_model():
    global _tokenizer, _nli_model

    if _tokenizer is None or _nli_model is None:
        print("Loading NLI model...")

        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)

        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL
        )

        _nli_model.eval()

    return _tokenizer, _nli_model


def _load_embedder():
    global _embedder

    if _embedder is None:
        print("Loading semantic embedding model...")

        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    return _embedder


# =========================================================
# NLI scoring
# =========================================================

def _nli_score(premise: str, hypothesis: str) -> dict:
    """
    Run Natural Language Inference.

    premise:
        Retrieved evidence.

    hypothesis:
        Answer claim.

    Returns:
        contradiction, entailment and neutral probabilities.
    """

    tokenizer, model = _load_nli_model()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    return {
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
    }


# =========================================================
# Semantic similarity
# =========================================================

def _semantic_similarity(
    claim: str,
    evidence_texts: list[str],
) -> float:
    """
    Calculate the maximum cosine similarity between a claim
    and the retrieved evidence.

    Returns:
        Float between 0 and 1.
    """

    if not claim or not evidence_texts:
        return 0.0

    embedder = _load_embedder()

    claim_embedding = embedder.encode(
        claim,
        normalize_embeddings=True,
    )

    evidence_embeddings = embedder.encode(
        evidence_texts,
        normalize_embeddings=True,
    )

    similarities = evidence_embeddings @ claim_embedding

    best_similarity = float(max(similarities))

    # Cosine similarity can theoretically be negative.
    # Normalize it into [0, 1].
    normalized_similarity = (best_similarity + 1.0) / 2.0

    return max(
        0.0,
        min(1.0, normalized_similarity),
    )


# =========================================================
# Sentence / claim splitting
# =========================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split an answer into individual claims/sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =========================================================
# Main faithfulness function
# =========================================================

def score_faithfulness(
    answer: str,
    evidence_chunks: list[dict],
    debug: bool = False,
) -> float:
    """
    Estimate how strongly the answer is supported by the
    retrieved evidence.

    The score combines:

        1. NLI entailment
        2. Semantic similarity
        3. Contradiction penalty

    Returns:
        Float between 0 and 1.
    """

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if not answer or not answer.strip():
        return 0.0

    if not evidence_chunks:
        return 0.0

    evidence_texts = [
        chunk.get("text", "").strip()
        for chunk in evidence_chunks
        if chunk.get("text")
    ]

    if not evidence_texts:
        return 0.0

    # -----------------------------------------------------
    # Split answer into claims
    # -----------------------------------------------------

    claims = _split_sentences(answer)

    if not claims:
        return 0.0

    claim_scores = []

    # -----------------------------------------------------
    # Evaluate every claim
    # -----------------------------------------------------

    for claim in claims:

        best_nli_entailment = 0.0
        best_nli_contradiction = 0.0
        best_evidence = ""

        # ---------------------------------------------
        # Find the strongest NLI evidence
        # ---------------------------------------------

        for evidence in evidence_texts:

            result = _nli_score(
                premise=evidence,
                hypothesis=claim,
            )

            if result["entailment"] > best_nli_entailment:

                best_nli_entailment = result["entailment"]

                best_nli_contradiction = result["contradiction"]

                best_evidence = evidence

        # ---------------------------------------------
        # Semantic similarity
        # ---------------------------------------------

        semantic_score = _semantic_similarity(
            claim,
            evidence_texts,
        )

        # ---------------------------------------------
        # Combine NLI + semantic similarity
        # ---------------------------------------------

        # NLI is the primary signal.
        #
        # Semantic similarity helps when the answer is a
        # valid paraphrase that NLI labels as "neutral".

        # -------------------------------------------------
        # Faithfulness decision
        # -------------------------------------------------

        if best_nli_contradiction >= 0.50:
        # Strong contradiction means the answer is not
        # supported by the evidence.
            combined_score = 0.0

        elif best_nli_entailment >= 0.50:
        # Strong NLI entailment is the strongest signal.
            combined_score = (
                0.80 * best_nli_entailment
                + 0.20 * semantic_score
            )

        else:
            # NLI considers the claim neutral.
            # Semantic similarity can rescue a legitimate
            # paraphrase, but only partially.
            combined_score = (
                0.20 * best_nli_entailment
                + 0.40 * semantic_score
            )

        # Clamp to [0, 1]
        combined_score = max(
        0.0,
        min(1.0, combined_score),
    )

        combined_score = max(
            0.0,
            min(1.0, combined_score),
        )

        claim_scores.append(combined_score)

        # ---------------------------------------------
        # Debug output
        # ---------------------------------------------

        if debug:

            print("\nClaim:")
            print(claim)

            print(
                f"Best NLI entailment : "
                f"{best_nli_entailment:.4f}"
            )

            print(
                f"Best contradiction  : "
                f"{best_nli_contradiction:.4f}"
            )

            print(
                f"Semantic similarity : "
                f"{semantic_score:.4f}"
            )

            print(
                f"Combined claim score: "
                f"{combined_score:.4f}"
            )

            print(
                f"Best evidence       : "
                f"{best_evidence[:200]}..."
            )

    # -----------------------------------------------------
    # Average support across claims
    # -----------------------------------------------------

    faithfulness = sum(claim_scores) / len(claim_scores)

    return round(
        max(0.0, min(1.0, faithfulness)),
        4,
    )