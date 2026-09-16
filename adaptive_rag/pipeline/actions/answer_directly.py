from .common import generate_answer

def answer_directly(query, chroma_collection, groq_client, state):
    """No retrieval — cheapest, fastest action. Good for simple factoid queries."""
    state["actions_taken"].append("answer_directly")
    answer = generate_answer(groq_client, query, evidence_chunks=[])
    return state, answer