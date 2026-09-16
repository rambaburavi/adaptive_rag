import os
from tavily import TavilyClient
from .common import generate_answer

_tavily = None

def get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily

def web_search(query, chroma_collection, groq_client, state, max_results=5):
    """Fallback to external search when the local corpus likely won't cover it."""
    tavily = get_tavily()
    try:
        results = tavily.search(query=query, max_results=max_results)
        evidence = [
            {"text": r["content"], "source": r["url"], "score": r.get("score", 0.0)}
            for r in results.get("results", [])
        ]
    except Exception as e:
        evidence = []
        state.setdefault("errors", []).append(f"web_search failed: {e}")

    state["evidence"].extend(evidence)
    state["actions_taken"].append("web_search")

    answer = generate_answer(groq_client, query, evidence)
    return state, answer