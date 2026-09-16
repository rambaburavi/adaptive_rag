import re

COMPARE_WORDS = {"compare", "vs", "versus", "difference between", "differ"}
RELATION_WORDS = {"related to", "relationship", "how does", "how are", "connect", "affect", "influence"}
DEFINITION_STARTERS = {"what is", "what are", "define", "meaning of"}

def extract_features(query: str) -> dict:
    q = query.strip()
    q_lower = q.lower()

    num_words = len(q.split())
    query_length = len(q)

    has_compare = any(w in q_lower for w in COMPARE_WORDS)
    has_relationship = any(w in q_lower for w in RELATION_WORDS)
    is_definition = any(q_lower.startswith(w) for w in DEFINITION_STARTERS)
    has_how = q_lower.startswith("how") or " how " in q_lower

    # crude concept count: capitalized-ish noun phrases / "and" splits as a proxy
    and_splits = len(re.split(r"\band\b", q_lower))
    num_concepts = max(1, and_splits)

    # simple heuristic complexity score, 0-1
    complexity_score = 0.0
    complexity_score += 0.3 if has_compare else 0.0
    complexity_score += 0.25 if has_relationship else 0.0
    complexity_score += 0.15 if has_how else 0.0
    complexity_score += min(0.3, 0.05 * max(0, num_concepts - 1))
    complexity_score += min(0.2, 0.01 * max(0, num_words - 8))
    complexity_score = min(1.0, complexity_score)

    return {
        "query_length": query_length,
        "num_words": num_words,
        "has_compare": has_compare,
        "has_relationship": has_relationship,
        "has_how": has_how,
        "is_definition": is_definition,
        "num_concepts": num_concepts,
        "complexity_score": round(complexity_score, 3),
    }