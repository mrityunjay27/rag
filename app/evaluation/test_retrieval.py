from app.evaluation.dataset import evaluation_data
from app.evaluation.recall import calculate_recall_at_k
from app.evaluation.precision import calculate_precision_at_k
from app.evaluation.reciprocal_rank import calculate_reciprocal_rank
from app.evaluation.context_relevance_metric import calculate_context_relevance


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

            context_relevance = calculate_context_relevance(
                query=query,
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
            print(f"Context Relevance: {context_relevance:.2%}")
        recall = correct / len(evaluation_data)
        precision = precision_sum / len(evaluation_data)
        mrr = reciprocal_rank_sum / len(evaluation_data)

        print("\n")
        print(f"Overall Recall@{k}: {recall:.2%}")
        print(f"Overall Precision@{k}: {precision:.2%}")
        print(f"MRR@{k}: {mrr:.2f}")
