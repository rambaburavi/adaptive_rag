from .common import get_embedder, generate_answer

def retrieve(query, chroma_collection, groq_client, state, k=5):
    """Baseline top-k dense retrieval, then answer from those chunks."""
    embedder = get_embedder()
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    results = chroma_collection.query(query_embeddings=[query_emb], n_results=k)

    evidence = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        evidence.append({"text": text, "source": meta.get("source", "unknown"), "score": 1 - dist})

    state["evidence"].extend(evidence)
    state["actions_taken"].append("retrieve")

    answer = generate_answer(groq_client, query, evidence)
    return state, answer