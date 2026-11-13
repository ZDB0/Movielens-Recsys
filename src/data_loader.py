import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

from src.config import (
    RATINGS_FILE,
    MOVIES_FILE,
    USERS_FILE,
    PROCESSED_DATA_DIR,
    TRAIN_MATRIX_PATH,
    TEST_MATRIX_PATH,
    MAPPINGS_PATH,
    MOVIES_CLEAN_PATH,
    TEST_RATIO,
)



def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Зчитує сирі .dat файли MovieLens 1M. 
    
    Returns:
        ratings (pd.DataFrame): DataFrame з оцінками користувачів.  
        movies (pd.DataFrame): DataFrame з інформацією про фільми.  
        users (pd.DataFrame): DataFrame з інформацією про користувачів.
    """
    ratings = pd.read_csv(
        RATINGS_FILE,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        MOVIES_FILE,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        USERS_FILE,
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="latin-1",
    )
    return ratings, movies, users


def temporal_train_test_split(
    ratings: pd.DataFrame, test_ratio: float = TEST_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ділить взаємодії кожного користувача за часом: перші (1 - test_ratio) у train, останні у test.

    Returns:
        train_df (pd.DataFrame): DataFrame для тренування.
        test_df (pd.DataFrame): DataFrame для тесту. 
    """
    ratings_sorted = ratings.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    user_counts = ratings_sorted.groupby("UserID")["MovieID"].transform("count")
    user_order = ratings_sorted.groupby("UserID").cumcount()

    test_mask = user_order >= (user_counts * (1.0 - test_ratio)).astype(int)

    train_df = ratings_sorted[~test_mask].copy()
    test_df = ratings_sorted[test_mask].copy()
    return train_df, test_df


def build_sparse_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[csr_matrix, csr_matrix, dict]:
    """
    Будує неперервні індекси та розріджені матриці User-Item (CSR).

    Returns:
        train_matrix (csr_matrix): Розріджена матриця для тренування
        test_matrix (csr_matrix): Розріджена матриця для тесту
        mappings (dict): Словник з відображенням індексів у UserID та MovieID
    """
    unique_users = sorted(train_df["UserID"].unique().tolist())
    unique_movies = sorted(train_df["MovieID"].unique().tolist())

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}

    # Відкидаємо з test рідкісні фільми, яких не було у train (уникнення cold-start при оцінці)
    test_filtered = test_df[
        test_df["UserID"].isin(user_to_idx) & test_df["MovieID"].isin(movie_to_idx)
    ].copy()

    n_users = len(unique_users)
    n_movies = len(unique_movies)

    train_rows = train_df["UserID"].map(user_to_idx).to_numpy()
    train_cols = train_df["MovieID"].map(movie_to_idx).to_numpy()
    train_vals = train_df["Rating"].to_numpy(dtype=np.float32)

    test_rows = test_filtered["UserID"].map(user_to_idx).to_numpy()
    test_cols = test_filtered["MovieID"].map(movie_to_idx).to_numpy()
    test_vals = test_filtered["Rating"].to_numpy(dtype=np.float32)

    train_matrix = csr_matrix(
        (train_vals, (train_rows, train_cols)), shape=(n_users, n_movies), dtype=np.float32
    )
    test_matrix = csr_matrix(
        (test_vals, (test_rows, test_cols)), shape=(n_users, n_movies), dtype=np.float32
    )

    mappings = {
        "idx_to_user": unique_users,
        "idx_to_movie": unique_movies,
    }
    return train_matrix, test_matrix, mappings


def prepare_and_save_data() -> None:
    """Повний пайплайн підготовки даних і збереження артефактів."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ratings, movies, _ = load_raw_data()
    train_df, test_df = temporal_train_test_split(ratings, TEST_RATIO)
    train_matrix, test_matrix, mappings = build_sparse_matrices(train_df, test_df)

    save_npz(TRAIN_MATRIX_PATH, train_matrix)
    save_npz(TEST_MATRIX_PATH, test_matrix)

    with open(MAPPINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(mappings, f)

    movies.to_csv(MOVIES_CLEAN_PATH, index=False)

    total_cells = train_matrix.shape[0] * train_matrix.shape[1]
    sparsity = 1.0 - (train_matrix.nnz / total_cells)

    print(f"Користувачів: {train_matrix.shape[0]} | Фільмів у train: {train_matrix.shape[1]}")
    print(f"Train взаємодій: {train_matrix.nnz:,} | Test взаємодій: {test_matrix.nnz:,}")
    print(f"Розрідженість (Sparsity) матриці: {sparsity:.2%}")
    print(f"Артефакти збережено у {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    prepare_and_save_data()