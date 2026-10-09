import pandas as pd

path = "data/queries/evaluation_queries.csv"
df = pd.read_csv(path)

replacements = {
    "How does clustering support unsupervised learning?":
        "How does data annotation support machine learning?",

    "How do recommender systems use machine learning?":
        "How does feature learning support machine learning?",

    "How does support vector machine classification work?":
        "How does dimensionality reduction support machine learning?",
}

for old, new in replacements.items():
    rows = df.index[df["query"] == old].tolist()

    if len(rows) != 2:
        raise ValueError(f"Expected exactly 2 copies of '{old}', found {len(rows)}")

    # Replace only the second occurrence.
    df.loc[rows[1], "query"] = new

df.to_csv(path, index=False)

print("Replacement complete.")
print("\nDistribution:")
print(df["expected_difficulty_label"].value_counts())

print("\nRemaining duplicate queries:")
dupes = df[df["query"].duplicated(keep=False)]
print(dupes[["query", "expected_difficulty_label"]].to_string(index=False))