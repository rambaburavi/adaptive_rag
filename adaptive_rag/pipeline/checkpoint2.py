import os
import sys
import pandas as pd
from dotenv import load_dotenv
import chromadb
from groq import Groq

sys.path.append(os.path.dirname(__file__))  # so 'actions' and 'feature_extractor' resolve

from actions import ACTIONS
from actions.common import new_state
from adaptive_router import adaptive_route

load_dotenv()

def main():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("wikipedia_rag")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    df = (
    pd.read_csv("data/queries/evaluation_queries.csv")
    .groupby("expected_difficulty_label", sort=False)
    .head(5)
    .reset_index(drop=True)
)  # sanity set
    results = []

    for _, row in df.iterrows():
        query = row["query"]
        expected_label = row["expected_difficulty_label"]  # ground truth — evaluation only, never routing

        action_name, features = adaptive_route(query)  # decision made purely from the query
        action_fn = ACTIONS[action_name]

        state = new_state()
        state, answer = action_fn(query, collection, groq_client, state)

        results.append({
            "query": query,
            "expected_label": expected_label,   # kept only for later evaluation comparison
            "chosen_action": action_name,
            "complexity_score": features["complexity_score"],
            "answer": answer,
            "num_evidence": len(state["evidence"]),
        })
        print(f"[expected={expected_label:10}] -> chosen={action_name:15} | {query}")

    pd.DataFrame(results).to_csv("eval/checkpoint2_adaptive_results.csv", index=False)
    print("\nSaved to eval/checkpoint2_adaptive_results.csv")

if __name__ == "__main__":
    main()