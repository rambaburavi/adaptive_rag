import os
import sys
import re
import pandas as pd
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

REFERENCES_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "queries",
    "full_references.csv",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "eval",
    "full_reference_validation.csv",
)

CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

COLLECTION_NAME = "wikipedia_rag"

TOP_K = 10


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_placeholder(text):
    """
    Detect references that say the corpus does not contain
    enough information instead of answering the question.
    """
    text_lower = text.lower()

    patterns = [
        "provided context does not contain",
        "does not contain any information",
        "context does not contain",
        "no information about",
        "no information on",
        "not enough information",
        "insufficient information",
        "cannot be answered from the context",
        "cannot answer from the context",
    ]

    return any(pattern in text_lower for pattern in patterns)


def word_count(text):
    return len(text.split())


def sentence_split(text):
    """
    Simple sentence splitter.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(query, collection, embedder, top_k=TOP_K):
    query_embedding = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    evidence = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        evidence.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "distance": float(distance),
                "similarity": max(
                    0.0,
                    min(1.0, 1.0 - float(distance))
                ),
            }
        )

    return evidence


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(text1, text2, embedder):
    emb1 = embedder.encode(
        text1,
        normalize_embeddings=True
    )

    emb2 = embedder.encode(
        text2,
        normalize_embeddings=True
    )

    similarity = float(emb1 @ emb2)

    # Convert [-1, 1] -> [0, 1]
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


# ============================================================
# CLAIM SUPPORT
# ============================================================

def calculate_claim_support(reference, evidence, embedder):
    """
    For every sentence/claim in the reference, find the most
    semantically similar retrieved evidence chunk.

    This is a screening heuristic, NOT a formal fact checker.
    """

    sentences = sentence_split(reference)

    if not sentences:
        return 0.0, []

    evidence_texts = [
        item["text"]
        for item in evidence
    ]

    claim_scores = []

    for sentence in sentences:
        best_score = 0.0
        best_source = "unknown"

        for item in evidence:
            score = semantic_similarity(
                sentence,
                item["text"],
                embedder
            )

            if score > best_score:
                best_score = score
                best_source = item["source"]

        claim_scores.append(
            {
                "claim": sentence,
                "support_score": round(best_score, 4),
                "source": best_source,
            }
        )

    average_support = sum(
        item["support_score"]
        for item in claim_scores
    ) / len(claim_scores)

    return average_support, claim_scores


# ============================================================
# NLI SCREENING
# ============================================================

def calculate_nli_support(reference, evidence, nli_model):
    """
    Checks whether retrieved evidence entails sentences from
    the reference.

    Uses the NLI model only as a screening signal.
    """

    sentences = sentence_split(reference)

    if not sentences or not evidence:
        return 0.0

    evidence_texts = [
        item["text"]
        for item in evidence
    ]

    entailment_scores = []

    for sentence in sentences:

        pairs = [
            [evidence_text, sentence]
            for evidence_text in evidence_texts
        ]

        predictions = nli_model.predict(
            pairs,
            show_progress_bar=False
        )

        best_entailment = 0.0

        for prediction in predictions:

            # Model labels:
            # 0 = contradiction
            # 1 = entailment
            # 2 = neutral

            if isinstance(prediction, (list, tuple)):
                if len(prediction) >= 3:
                    entailment = float(prediction[1])
                else:
                    continue
            else:
                continue

            best_entailment = max(
                best_entailment,
                entailment
            )

        entailment_scores.append(best_entailment)

    if not entailment_scores:
        return 0.0

    return sum(entailment_scores) / len(entailment_scores)


# ============================================================
# LENGTH CHECK
# ============================================================

def length_status(count):
    if count < 15:
        return "too_short"

    if count > 300:
        return "too_long"

    return "normal"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL REFERENCE VALIDATION")
    print("=" * 70)

    print("\nLoading references...")

    df = pd.read_csv(REFERENCES_FILE)

    print(f"References loaded: {len(df)}")

    required_columns = {
        "query",
        "reference_answer",
        "review_status",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\nLoading ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Chroma collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents in collection: "
        f"{collection.count()}"
    )

    print("\nLoading embedding model...")

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("\nLoading NLI model...")

    nli_model = CrossEncoder(
        NLI_MODEL
    )

    print("\nStarting validation...\n")

    results = []

    for position, row in df.iterrows():

        query = clean_text(row["query"])
        reference = clean_text(
            row["reference_answer"]
        )

        print(
            f"[{position + 1:03d}/"
            f"{len(df):03d}] "
            f"{query}"
        )

        # ----------------------------------------------------
        # Basic checks
        # ----------------------------------------------------

        words = word_count(reference)

        placeholder = is_placeholder(
            reference
        )

        length = length_status(words)

        empty = len(reference) == 0

        # ----------------------------------------------------
        # Retrieve corpus evidence
        # ----------------------------------------------------

        if not empty:

            evidence = retrieve_evidence(
                query,
                collection,
                embedder,
                TOP_K
            )

        else:
            evidence = []

        # ----------------------------------------------------
        # Retrieval relevance
        # ----------------------------------------------------

        if evidence:

            top_similarity = evidence[0][
                "similarity"
            ]

            average_similarity = sum(
                item["similarity"]
                for item in evidence
            ) / len(evidence)

        else:

            top_similarity = 0.0
            average_similarity = 0.0

        # ----------------------------------------------------
        # Reference → evidence semantic support
        # ----------------------------------------------------

        if reference and evidence:

            claim_support, claim_details = (
                calculate_claim_support(
                    reference,
                    evidence,
                    embedder
                )
            )

        else:

            claim_support = 0.0
            claim_details = []

        # ----------------------------------------------------
        # NLI support
        # ----------------------------------------------------

        if reference and evidence:

            nli_support = calculate_nli_support(
                reference,
                evidence,
                nli_model
            )

        else:

            nli_support = 0.0

        # ----------------------------------------------------
        # Determine validation status
        # ----------------------------------------------------

        reasons = []

        if empty:
            reasons.append("empty_reference")

        if placeholder:
            reasons.append("placeholder_reference")

        if length == "too_short":
            reasons.append("too_short")

        if length == "too_long":
            reasons.append("too_long")

        if top_similarity < 0.35:
            reasons.append(
                "weak_query_corpus_match"
            )

        if claim_support < 0.45:
            reasons.append(
                "weak_reference_support"
            )

        if nli_support < 0.40:
            reasons.append(
                "weak_nli_support"
            )

        # ----------------------------------------------------
        # Final screening category
        # ----------------------------------------------------

        if (
            empty
            or placeholder
            or top_similarity < 0.35
            or claim_support < 0.45
            or nli_support < 0.40
        ):
            validation_status = "needs_review"

        elif (
            length == "too_short"
            or length == "too_long"
        ):
            validation_status = "needs_review"

        else:
            validation_status = "pass"

        # ----------------------------------------------------
        # Top evidence sources
        # ----------------------------------------------------

        top_sources = "; ".join(
            dict.fromkeys(
                item["source"]
                for item in evidence[:5]
            )
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "row_index": position,
                "query": query,
                "reference_answer": reference,
                "word_count": words,
                "placeholder": placeholder,
                "length_status": length,
                "top_corpus_similarity": round(
                    top_similarity,
                    4
                ),
                "average_corpus_similarity": round(
                    average_similarity,
                    4
                ),
                "reference_claim_support": round(
                    claim_support,
                    4
                ),
                "nli_entailment_support": round(
                    nli_support,
                    4
                ),
                "validation_status": validation_status,
                "reasons": "; ".join(reasons),
                "top_sources": top_sources,
                "original_review_status": clean_text(
                    row["review_status"]
                ),
            }
        )

    # ========================================================
    # SAVE
    # ========================================================

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Total references : {len(result_df)}"
    )

    print("\nValidation status:")

    print(
        result_df[
            "validation_status"
        ].value_counts()
    )

    print("\nPlaceholder references:")

    print(
        result_df[
            result_df["placeholder"]
        ][
            [
                "row_index",
                "query",
                "reference_answer",
            ]
        ].to_string(index=False)
    )

    print("\nNeeds review:")

    review_df = result_df[
        result_df["validation_status"]
        == "needs_review"
    ]

    print(
        review_df[
            [
                "row_index",
                "query",
                "word_count",
                "top_corpus_similarity",
                "reference_claim_support",
                "nli_entailment_support",
                "reasons",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nNeeds review count: "
        f"{len(review_df)}"
    )

    print(
        f"\nValidation report saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()