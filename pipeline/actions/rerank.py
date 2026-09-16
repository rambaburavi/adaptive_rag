from sentence_transformers import CrossEncoder
from .common import get_embedder, generate_answer

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def rerank(query, chroma_collection, groq_client, state, initial_k=10, final_k=5):
    """Pull a wider candidate pool, rerank with a cross-encoder, keep top final_k."""
    embedder = get_embedder()
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    results = chroma_collection.query(query_embeddings=[query_emb], n_results=initial_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        state["actions_taken"].append("rerank")
        answer = generate_answer(groq_client, query, [])
        return state, answer

    cross_encoder = get_cross_encoder()
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)[:final_k]

    evidence = [{"text": text, "source": meta.get("source", "unknown"), "score": float(score)}
                for text, meta, score in ranked]

    state["evidence"].extend(evidence)
    state["actions_taken"].append("rerank")

    answer = generate_answer(groq_client, query, evidence)
    return state, answer