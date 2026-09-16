import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Configuration
# ============================================================

CORPUS_DIR = Path("data/corpus")
CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "wikipedia_rag"

# Roughly 300–500 tokens.
# 1 token is approximately 4 characters for English text.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Load embedding model
# ============================================================

print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Embedding model: {EMBEDDING_MODEL}")
print("Model loaded successfully.")


# ============================================================
# Create text splitter
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ============================================================
# Create ChromaDB
# ============================================================

print("\nCreating ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(f"Collection: {COLLECTION_NAME}")


# ============================================================
# Read Wikipedia articles
# ============================================================

json_files = list(CORPUS_DIR.glob("*.json"))

print(f"\nFound {len(json_files)} Wikipedia articles.")


all_documents = []
all_embeddings = []
all_metadatas = []
all_ids = []


# ============================================================
# Process articles
# ============================================================

for article_number, json_file in enumerate(json_files, start=1):

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            article = json.load(f)

        title = article.get("title", "")
        page_id = article.get("pageid", "")
        url = article.get("url", "")
        text = article.get("text", "")

        if not text.strip():
            continue

        # Split article into chunks
        chunks = text_splitter.split_text(text)

        print(
            f"[{article_number}/{len(json_files)}] "
            f"{title} → {len(chunks)} chunks"
        )

        for chunk_number, chunk in enumerate(chunks):

            document_id = f"{page_id}_{chunk_number}"

            metadata = {
                "source": title,
                "page_id": str(page_id),
                "url": url,
                "chunk_id": chunk_number
            }

            all_documents.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(document_id)

    except Exception as e:

        print(f"Error processing {json_file}: {e}")


# ============================================================
# Generate embeddings
# ============================================================

print("\n" + "=" * 60)
print("Generating embeddings...")
print("=" * 60)

print(f"Total chunks: {len(all_documents)}")

all_embeddings = model.encode(
    all_documents,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)

print("Embeddings generated successfully.")


# ============================================================
# Store in ChromaDB
# ============================================================

print("\n" + "=" * 60)
print("Storing chunks in ChromaDB...")
print("=" * 60)

# Add in batches to avoid excessive memory/API payload size
BATCH_SIZE = 500

for start in range(0, len(all_documents), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(all_documents))

    collection.add(
        ids=all_ids[start:end],
        documents=all_documents[start:end],
        embeddings=all_embeddings[start:end].tolist(),
        metadatas=all_metadatas[start:end]
    )

    print(
        f"Stored {end}/{len(all_documents)} chunks"
    )


# ============================================================
# Final verification
# ============================================================

print("\n" + "=" * 60)
print("VECTOR DATABASE BUILD COMPLETE")
print("=" * 60)

print(f"Articles processed : {len(json_files)}")
print(f"Total chunks       : {len(all_documents)}")
print(f"ChromaDB location  : {CHROMA_DIR.resolve()}")
print(f"Collection         : {COLLECTION_NAME}")
print(f"Database count     : {collection.count()}")

print("\nStep 3 completed successfully!")