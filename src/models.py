import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from implicit.als import AlternatingLeastSquares

from src.config import (
    N_FACTORS,
    ALS_REGULARIZATION,
    ALS_ITERATIONS,
    ALS_ALPHA,
    RANDOM_STATE,
)



class SVDRecommender:
    """Колаборативна фільтрація на базі централізованого Truncated SVD (SciPy)."""

    def __init__(self, n_factors: int = N_FACTORS, random_state: int = RANDOM_STATE):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_means: np.ndarray | None = None

    def fit(self, train_matrix: csr_matrix) -> "SVDRecommender":
        # Обчислюємо середній рейтинг кожного користувача лише по відомих оцінках
        row_sums = np.asarray(train_matrix.sum(axis=1)).ravel()
        row_counts = np.diff(train_matrix.indptr)
        self.user_means = np.divide(
            row_sums,
            row_counts,
            out=np.zeros_like(row_sums, dtype=np.float32),
            where=row_counts > 0,
        )

        # Центруємо матрицю без перетворення у dense-формат
        centered = train_matrix.copy().astype(np.float32)
        centered.data -= np.repeat(self.user_means, row_counts)

        u, s, vt = svds(centered, k=self.n_factors, random_state=self.random_state)

        # svds повертає сингулярні числа за зростанням — розвертаємо у спадному порядку
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]

        self.user_factors = (u * s).astype(np.float32)
        self.item_factors = vt.T.astype(np.float32)
        return self

    def recommend(
        self, user_ids: np.ndarray, train_matrix: csr_matrix, k: int = 10
    ) -> np.ndarray:
        scores = (
            self.user_factors[user_ids] @ self.item_factors.T
            + self.user_means[user_ids, None]
        )

        # Маскуємо фільми, які користувач уже дивився у train
        seen_mask = train_matrix[user_ids].toarray() > 0
        scores[seen_mask] = -np.inf

        top_k_part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(len(user_ids))[:, None]
        top_k_scores = scores[row_idx, top_k_part]
        sorted_order = np.argsort(-top_k_scores, axis=1)

        return top_k_part[row_idx, sorted_order]


class ALSRecommender:
    """Колаборативна фільтрація на базі Alternating Least Squares (бібліотека implicit)."""

    def __init__(
        self,
        factors: int = N_FACTORS,
        regularization: float = ALS_REGULARIZATION,
        iterations: int = ALS_ITERATIONS,
        alpha: float = ALS_ALPHA,
        random_state: int = RANDOM_STATE,
    ):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
        )

    def fit(self, train_matrix: csr_matrix, min_rating: float = 4.0) -> "ALSRecommender":
        # Для implicit ALS залишаємо позитивні взаємодії (оцінка >= 4.0)
        positive_matrix = train_matrix.copy().astype(np.float32)
        positive_matrix.data = np.where(positive_matrix.data >= min_rating, 1.0, 0.0)
        positive_matrix.eliminate_zeros()

        self.model.fit(positive_matrix)
        return self

    @property
    def item_factors(self) -> np.ndarray:
        return np.asarray(self.model.item_factors, dtype=np.float32)

    @property
    def user_factors(self) -> np.ndarray:
        return np.asarray(self.model.user_factors, dtype=np.float32)

    def recommend(
        self, user_ids: np.ndarray, train_matrix: csr_matrix, k: int = 10
    ) -> np.ndarray:
        # Передаємо повну train_matrix у filter_already_liked_items,
        # щоб виключити всі раніше оцінені фільми (включно з оцінками 1-3)
        rec_ids, _ = self.model.recommend(
            userid=user_ids,
            user_items=train_matrix[user_ids],
            N=k,
            filter_already_liked_items=True,
        )
        return rec_ids