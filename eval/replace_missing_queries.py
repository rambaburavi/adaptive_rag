import pandas as pd

EVAL_FILE = "data/queries/evaluation_queries.csv"
REF_FILE = "data/queries/full_references.csv"

replacements = {
    "How does boosting combine weak learners?":
        "How does clustering support unsupervised learning?",

    "How does tokenization prepare text for language models?":
        "How do recommender systems use machine learning?",

    "Compare traditional machine learning and deep learning in terms of data requirements.":
        "Compare biological neurons and artificial neurons.",

    "Compare overfitting and underfitting.":
        "Compare decision trees and random forests.",
}

eval_df = pd.read_csv(EVAL_FILE)
ref_df = pd.read_csv(REF_FILE)

# Replace the four unsupported evaluation queries.
eval_df["query"] = eval_df["query"].replace(replacements)

# Remove the four old unsupported queries from references.
old_queries = set(replacements.keys())

ref_df = ref_df[
    ~ref_df["query"].isin(old_queries)
].copy()

# The replacement references already exist in the file.
# Remove any accidental duplicate rows just in case.
ref_df = ref_df.drop_duplicates(
    subset=["query"],
    keep="first"
).copy()

# Make sure every replacement query has a reference.
replacement_queries = list(replacements.values())

missing_replacements = [
    q for q in replacement_queries
    if q not in set(ref_df["query"])
]

if missing_replacements:
    raise ValueError(
        "Missing replacement references:\n"
        + "\n".join(missing_replacements)
    )

# Rebuild the reference file in exactly the same order
# as the evaluation query file.
ref_map = ref_df.set_index("query").to_dict(orient="index")

rows = []

for query in eval_df["query"]:
    if query not in ref_map:
        raise ValueError(
            f"Missing reference for evaluation query: {query}"
        )

    rows.append({
        "query": query,
        "reference_answer": ref_map[query]["reference_answer"],
        "review_status": ref_map[query]["review_status"],
    })

final_ref_df = pd.DataFrame(rows)

# Final safety checks.
if len(eval_df) != 200:
    raise ValueError(
        f"Expected 200 evaluation queries, found {len(eval_df)}"
    )

if len(final_ref_df) != 200:
    raise ValueError(
        f"Expected 200 references, found {len(final_ref_df)}"
    )

if final_ref_df["query"].duplicated().any():
    raise ValueError("Duplicate queries found in final reference file.")

missing_mask = (
    final_ref_df["reference_answer"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
)

if missing_mask.any():
    raise ValueError(
        f"{missing_mask.sum()} references are still empty."
    )

# Save both files.
eval_df.to_csv(EVAL_FILE, index=False)
final_ref_df.to_csv(REF_FILE, index=False)

print("========================================")
print("REPLACEMENT COMPLETE")
print("========================================")
print(f"Evaluation queries : {len(eval_df)}")
print(f"References         : {len(final_ref_df)}")
print(f"Missing references : {missing_mask.sum()}")
print()
print("Difficulty distribution:")
print(eval_df["expected_difficulty_label"].value_counts())
print()
print("Reference status:")
print(final_ref_df["review_status"].value_counts())
print()
print("Replacements:")
for old, new in replacements.items():
    print(f"- {old}")
    print(f"  -> {new}")