
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from reranker import PassageReranker


PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "tnnmc_knowledge_base"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Retrieve more candidates before reranking.
RETRIEVAL_TOP_K = 15

# Number of final passages returned.
FINAL_TOP_K = 3


class KnowledgeBaseRetriever:
    def __init__(self):
        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu"
        )

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print("Loading cross-encoder reranker...")

        self.reranker = PassageReranker()

        print("Knowledge base retriever ready.")

    def search(self, question, retrieval_top_k=RETRIEVAL_TOP_K,
               final_top_k=FINAL_TOP_K):
        """Retrieve candidate chunks and rerank them."""

        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(
                retrieval_top_k,
                self.collection.count()
            )
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        candidates = [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            }
            for i in range(len(documents))
        ]

        reranked_results = self.reranker.rerank(
            question=question,
            passages=candidates,
            top_k=final_top_k
        )

        return reranked_results


if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever()

    questions = [
        "What career opportunities are available after GNM?",
        "What higher study options are available for nurses?",
        "GNM முடித்த பிறகு என்ன வேலை வாய்ப்புகள் உள்ளன?"
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = retriever.search(question)

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Record ID:", result["id"])
            print(
                "Source:",
                result["metadata"].get("source_file")
            )
            print(
                "Record type:",
                result["metadata"].get("record_type")
            )
            print(
                "Original cosine distance:",
                round(result["distance"], 4)
            )
            print(
                "Reranker score:",
                round(result["reranker_score"], 4)
            )
            print("Text:")
            print(result["text"][:600])
