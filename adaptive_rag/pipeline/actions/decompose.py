from .retrieve import retrieve
from .common import generate_answer, GROQ_MODEL

def _split_into_subquestions(query, groq_client, max_subs=3):
    prompt = (
        f"Break the following question into {max_subs} or fewer simpler sub-questions "
        f"that together would help answer it. Return only the sub-questions, one per line, "
        f"no numbering.\n\nQuestion: {query}"
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    lines = [l.strip("-• ").strip() for l in resp.choices[0].message.content.strip().split("\n")]
    return [l for l in lines if l][:max_subs]

def decompose(query, chroma_collection, groq_client, state):
    """Split complex/comparison queries into sub-questions, retrieve for each, then synthesize."""
    sub_questions = _split_into_subquestions(query, groq_client)

    for sub_q in sub_questions:
        state, sub_answer = retrieve(sub_q, chroma_collection, groq_client, state)
        state["sub_answers"].append({"question": sub_q, "answer": sub_answer})

    state["actions_taken"].append("decompose")

    synthesis_context = "\n\n".join(
        f"Sub-question: {sa['question']}\nSub-answer: {sa['answer']}" for sa in state["sub_answers"]
    )
    prompt = (
        f"Using the sub-answers below, give one combined answer to the original question.\n\n"
        f"{synthesis_context}\n\nOriginal question: {query}\n\nCombined answer:"
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    final_answer = resp.choices[0].message.content.strip()
    return state, final_answer