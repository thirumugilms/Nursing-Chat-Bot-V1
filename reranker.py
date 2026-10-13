
import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class PassageReranker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading cross-encoder on {self.device.upper()}...")

        self.model = CrossEncoder(
            MODEL_NAME,
            max_length=512,
            device=self.device
        )

        print("Cross-encoder loaded successfully.")

    def rerank(self, question, passages, top_k=5):
        if not passages:
            return []

        pairs = [
            (question, passage["text"])
            for passage in passages
        ]

        scores = self.model.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False
        )

        ranked_results = []

        for passage, score in zip(passages, scores):
            result = passage.copy()
            result["reranker_score"] = float(score)
            ranked_results.append(result)

        ranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return ranked_results[:top_k]


if __name__ == "__main__":
    reranker = PassageReranker()

    question = "What career opportunities are available after GNM?"

    sample_passages = [
        {
            "id": "example_1",
            "text": (
                "Higher study information about M.Sc Nursing "
                "and postgraduate career progression."
            ),
            "metadata": {"source_file": "higher_study.json"}
        },
        {
            "id": "example_2",
            "text": (
                "Qualification: General Nursing and Midwifery (GNM). "
                "Career roles include Staff Nurse, Ward Nurse, "
                "ICU Nurse, Emergency Nurse and Community Health Nurse."
            ),
            "metadata": {"source_file": "career.json"}
        },
        {
            "id": "example_3",
            "text": (
                "Career growth pathways and promotion opportunities "
                "for nursing professionals."
            ),
            "metadata": {"source_file": "jobs.json"}
        }
    ]

    results = reranker.rerank(
        question,
        sample_passages,
        top_k=3
    )

    print("\nQuestion:", question)
    print("\nReranked results:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("ID:", result["id"])
        print("Source:", result["metadata"]["source_file"])
        print(
            "Reranker score:",
            round(result["reranker_score"], 4)
        )
        print("Text:", result["text"])
