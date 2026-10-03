from app.evaluation.dataset import evaluation_data
from app.retrieval.search import search


def calculate_recall_at_k(
    query: str,
    expected_evidence: str,
    k: int,
) -> int:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    for content, source, chunk_index, similarity in results:

        if expected_evidence.lower() in content.lower():
            return 1

    return 0


def calculate_precision_at_k(
    query: str,
    expected_evidence: str,
    k: int,
) -> float:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    if not results:
        return 0.0

    relevant_chunks = 0

    for content, source, chunk_index, similarity in results:

        if expected_evidence.lower() in content.lower():
            relevant_chunks += 1

    return relevant_chunks / len(results)

def calculate_reciprocal_rank(
    query: str,
    expected_evidence: str,
    k: int,
) -> float:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    for rank, (
        content,
        source,
        chunk_index,
        similarity,
    ) in enumerate(results, start=1):

        if expected_evidence.lower() in content.lower():
            return 1 / rank

    return 0.0

if __name__ == "__main__":

    for k in [1, 2]:

        correct = 0
        precision_sum = 0.0
        reciprocal_rank_sum = 0.0

        print("\n")
        print("=" * 80)
        print(f"RETRIEVAL EVALUATION @ {k}")
        print("=" * 80)

        for item in evaluation_data:

            query = item["query"]
            expected_evidence = item["expected_evidence"]

            recall = calculate_recall_at_k(
                query=query,
                expected_evidence=expected_evidence,
                k=k,
            )

            precision = calculate_precision_at_k(
                query=query,
                expected_evidence=expected_evidence,
                k=k,
            )

            reciprocal_rank = calculate_reciprocal_rank(
                query=query,
                expected_evidence=expected_evidence,
                k=k,
            )

            correct += recall
            precision_sum += precision
            reciprocal_rank_sum += reciprocal_rank

            print(f"\nQuery: {query}")
            print(f"Expected evidence: {expected_evidence}")
            print(f"Recall@{k}: {'YES' if recall else 'NO'}")
            print(f"Precision@{k}: {precision:.2%}")
            print(f"Reciprocal Rank: {reciprocal_rank:.2f}")

        recall = correct / len(evaluation_data)
        precision = precision_sum / len(evaluation_data)
        mrr = reciprocal_rank_sum / len(evaluation_data)

        print("\n")
        print(f"Overall Recall@{k}: {recall:.2%}")
        print(f"Overall Precision@{k}: {precision:.2%}")
        print(f"MRR@{k}: {mrr:.2f}")