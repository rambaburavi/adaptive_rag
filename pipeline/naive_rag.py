import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "wikipedia_rag"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Make sure it is present in .env"
    )


# ============================================================
# Initialize Groq
# ============================================================

groq_client = Groq(api_key=api_key)


# ============================================================
# Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# ============================================================
# Connect to ChromaDB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(f"ChromaDB collection loaded.")
print(f"Documents in collection: {collection.count()}")


# ============================================================
# Retrieve relevant chunks
# ============================================================

def retrieve_documents(query, top_k=TOP_K):

        # Convert user query into an embedding
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return documents, metadatas, distances


    # ============================================================
# Build prompt
# ============================================================

def build_prompt(query, documents, metadatas):

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        source = metadata.get("source", "Unknown")

        context_parts.append(
            f"""
SOURCE {i}
Article: {source}

{document}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the provided
Wikipedia context.

If the context does not contain enough information to
answer the question, say that the provided context
does not contain enough information.

Do not invent facts that are not supported by the context.

USER QUESTION:
{query}

WIKIPEDIA CONTEXT:
{context}

Provide a clear and concise answer.
"""

    return prompt


# ============================================================
# Generate answer using Groq
# ============================================================

def generate_answer(prompt):

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reliable RAG question-answering "
                    "assistant. Use only the supplied context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# Complete RAG pipeline
# ============================================================

def answer_query(query):

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    # Step 1: Retrieve
    documents, metadatas, distances = retrieve_documents(query)

    print("\n" + "=" * 70)
    print(f"RETRIEVED TOP {len(documents)} CHUNKS")
    print("=" * 70)

    for i, (metadata, distance) in enumerate(
        zip(metadatas, distances),
        start=1
    ):

        print(
            f"{i}. {metadata.get('source', 'Unknown')} "
            f"(distance: {distance:.4f})"
        )

    # Step 2: Build prompt
    prompt = build_prompt(
        query,
        documents,
        metadatas
    )

    # Step 3: Generate answer
    answer = generate_answer(prompt)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    return answer


# ============================================================
# Interactive mode
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("NAIVE RAG PIPELINE")
    print("=" * 70)

    print("Type a question.")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nAsk a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            continue

        try:
            answer_query(query)

        except Exception as e:
            print(f"\nError: {e}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()