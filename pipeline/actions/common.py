from sentence_transformers import SentenceTransformer

_embedder = None

GROQ_MODEL = "openai/gpt-oss-20b"  # same model as naive baseline — keep experiment fair

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def new_state():
    """Fresh state dict every module reads/writes."""
    return {
        "evidence": [],       # list of {"text":, "source":, "score":}
        "actions_taken": [],  # e.g. ["retrieve", "rerank"]
        "sub_answers": [],    # used by decompose
    }

def generate_answer(groq_client, query, evidence_chunks, model=GROQ_MODEL):
    """Shared Groq call: build a context prompt from evidence and answer."""
    if evidence_chunks:
        context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(evidence_chunks))
        prompt = (
            f"Answer the question using the context below. "
            f"If the context doesn't fully answer it, say what's missing.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
    else:
        prompt = f"Answer the question directly using your own knowledge.\n\nQuestion: {query}\n\nAnswer:"

    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()