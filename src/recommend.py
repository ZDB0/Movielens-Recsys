import argparse
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.config import (
    TRAIN_MATRIX_PATH,
    MAPPINGS_PATH,
    MOVIES_CLEAN_PATH,
    TOP_K,
)
from src.data_loader import prepare_and_save_data
from src.models import ALSRecommender
from src.fast_similarity import top_k_similar_items



def run_recommendation_demo(user_id: int, movie_id: int, top_k: int = TOP_K) -> None:
    if not TRAIN_MATRIX_PATH.exists() or not MAPPINGS_PATH.exists():
        print("Підготовлені матриці не знайдено. Запускаю підготовку даних...")
        prepare_and_save_data()

    train_matrix = load_npz(TRAIN_MATRIX_PATH)
    movies_df = pd.read_csv(MOVIES_CLEAN_PATH)
    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    idx_to_user = mappings["idx_to_user"]
    idx_to_movie = mappings["idx_to_movie"]
    user_to_idx = {uid: idx for idx, uid in enumerate(idx_to_user)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(idx_to_movie)}
    movie_lookup = movies_df.set_index("MovieID")[["Title", "Genres"]].to_dict("index")

    if user_id not in user_to_idx:
        raise ValueError(f"Користувача з UserID={user_id} не знайдено в базі.")
    if movie_id not in movie_to_idx:
        raise ValueError(f"Фільм із MovieID={movie_id} не знайдено в базі.")

    print("Навчання фінальної моделі ALS...")
    model = ALSRecommender().fit(train_matrix)

    # 1. Персональні рекомендації для користувача
    u_idx = user_to_idx[user_id]
    liked_indices = train_matrix[u_idx].indices[train_matrix[u_idx].data == 5.0][:5]

    print(f"\n=== Персональні рекомендації для UserID={user_id} ===")
    print("Приклади улюблених фільмів в історії (оцінка 5):")
    for idx in liked_indices:
        info = movie_lookup[idx_to_movie[int(idx)]]
        print(f"  * {info['Title']} [{info['Genres']}]")

    rec_indices = model.recommend(np.array([u_idx]), train_matrix, k=top_k)[0]
    print(f"\nТоп-{top_k} персональних рекомендацій (ALS):")
    for rank, idx in enumerate(rec_indices, 1):
        info = movie_lookup[idx_to_movie[int(idx)]]
        print(f"  {rank}. {info['Title']} [{info['Genres']}]")

    # 2. Пошук схожих фільмів через C++ (pybind11)
    m_idx = movie_to_idx[movie_id]
    item_embeddings = np.ascontiguousarray(model.item_factors, dtype=np.float32)
    sim_indices, sim_scores = top_k_similar_items(item_embeddings, query_idx=m_idx, k=top_k)

    target_info = movie_lookup[movie_id]
    print(f"\n=== Схожі фільми до '{target_info['Title']}' [{target_info['Genres']}] (C++ pybind11) ===")
    for rank, (idx, score) in enumerate(zip(sim_indices, sim_scores), 1):
        info = movie_lookup[idx_to_movie[int(idx)]]
        print(f"  {rank}. {info['Title']} [{info['Genres']}] — cosine similarity: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерація рекомендацій MovieLens 1M (ALS + C++)")
    parser.add_argument("--user-id", type=int, default=1, help="ID користувача з MovieLens (1..6040)")
    parser.add_argument("--movie-id", type=int, default=1, help="ID фільму для пошуку схожих (за замовчуванням 1 = Toy Story)")
    parser.add_argument("--top-k", type=int, default=5, help="Кількість рекомендацій")
    args = parser.parse_args()

    run_recommendation_demo(user_id=args.user_id, movie_id=args.movie_id, top_k=args.top_k)