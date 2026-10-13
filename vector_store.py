from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from text_processor import split_records
from data_loader import load_all_records


# Configuration
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "tnnmc_knowledge_base"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def build_vector_store():
    """Create embeddings and save them in ChromaDB."""

    print("Loading source records...")
    records = load_all_records()

    print("\nSplitting records into chunks...")
    chunks = split_records(records)

    print(f"Total chunks: {len(chunks)}")

    print("\nLoading multilingual embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Rebuild the collection when this script is intentionally rerun.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("\nCreating embeddings and storing chunks...")

    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["id"] for chunk in batch]

        metadatas = [
            {
                key: value
                for key, value in chunk["metadata"].items()
                if isinstance(value, (str, int, float, bool))
            }
            for chunk in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        completed = min(start + batch_size, len(chunks))
        print(f"Stored {completed}/{len(chunks)} chunks")

    print("\n" + "=" * 50)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 50)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Database directory: {CHROMA_DIR}")

    return collection


if __name__ == "__main__":
    build_vector_store()