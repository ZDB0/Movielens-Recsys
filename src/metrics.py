import numpy as np
from scipy.sparse import csr_matrix



def precision_at_k(recommended_ids: np.ndarray, relevant_set: set[int], k: int) -> float:
    """Обчислює Precision@K для одного користувача."""
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_set)
    return hits / k


def ndcg_at_k(recommended_ids: np.ndarray, relevant_set: set[int], k: int) -> float:
    """Обчислює NDCG@K (з бінарною релевантністю) для одного користувача."""
    if not relevant_set or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for rank, item_id in enumerate(top_k):
        if item_id in relevant_set:
            dcg += 1.0 / np.log2(rank + 2)  # rank починається з 0 -> log2(2) = 1.0

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_recommendations(
    user_ids: np.ndarray,
    recommendations: np.ndarray,
    test_matrix: csr_matrix,
    k: int = 10,
    min_rating: float = 4.0,
) -> dict[str, float]:
    """
    Оцінює середні Precision@K та NDCG@K по всіх користувачах,
    у яких у test_matrix є хоча б один релевантний фільм (оцінка >= min_rating).
    """
    precisions = []
    ndcgs = []

    for idx, user_id in enumerate(user_ids):
        row = test_matrix[user_id]
        relevant_items = set(row.indices[row.data >= min_rating].tolist())
        if not relevant_items:
            continue

        rec_ids = recommendations[idx]
        precisions.append(precision_at_k(rec_ids, relevant_items, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant_items, k))

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "evaluated_users": len(precisions),
    }