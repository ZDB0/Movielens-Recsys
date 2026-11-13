#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>



namespace py = pybind11;



// Допоміжна функція: обчислює L2-норми кожного рядка матриці (N x D)
std::vector<float> compute_row_norms(const py::detail::unchecked_reference<float, 2>& mat) {
    const py::ssize_t n_items = mat.shape(0);
    const py::ssize_t dim = mat.shape(1);
    std::vector<float> norms(n_items, 0.0f);

    for (py::ssize_t i = 0; i < n_items; ++i) {
        float sum_sq = 0.0f;
        for (py::ssize_t d = 0; d < dim; ++d) {
            const float val = mat(i, d);
            sum_sq += val * val;
        }
        norms[i] = std::sqrt(sum_sq);
    }
    return norms;
}

// 1. Пошук Top-K схожих фільмів для одного заданого фільму (query_idx)
py::tuple top_k_similar_items(
    py::array_t<float, py::array::c_style | py::array::forcecast> embeddings,
    py::ssize_t query_idx,
    int k
) {
    if (embeddings.ndim() != 2) {
        throw std::runtime_error("Очікується двовимірний масив (N, D)");
    }

    auto mat = embeddings.unchecked<2>();
    const py::ssize_t n_items = mat.shape(0);
    const py::ssize_t dim = mat.shape(1);

    if (query_idx < 0 || query_idx >= n_items) {
        throw std::out_of_range("query_idx виходить за межі кількості об'єктів");
    }

    const int safe_k = std::min(std::max(k, 1), static_cast<int>(n_items) - 1);
    const std::vector<float> norms = compute_row_norms(mat);
    const float query_norm = norms[query_idx];

    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_items - 1);

    for (py::ssize_t j = 0; j < n_items; ++j) {
        if (j == query_idx) {
            continue;
        }
        float sim = 0.0f;
        if (query_norm > 1e-12f && norms[j] > 1e-12f) {
            float dot = 0.0f;
            for (py::ssize_t d = 0; d < dim; ++d) {
                dot += mat(query_idx, d) * mat(j, d);
            }
            sim = dot / (query_norm * norms[j]);
        }
        scored.emplace_back(sim, static_cast<int>(j));
    }

    // Часткове сортування O(N log K) замість повного O(N log N)
    std::partial_sort(
        scored.begin(),
        scored.begin() + safe_k,
        scored.end(),
        [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        }
    );

    py::array_t<int> out_indices(safe_k);
    py::array_t<float> out_scores(safe_k);
    auto idx_ptr = out_indices.mutable_unchecked<1>();
    auto scr_ptr = out_scores.mutable_unchecked<1>();

    for (int i = 0; i < safe_k; ++i) {
        idx_ptr(i) = scored[i].second;
        scr_ptr(i) = scored[i].first;
    }

    return py::make_tuple(out_indices, out_scores);
}

// 2. Побудова таблиці Top-K схожих фільмів одразу для ВСІХ N фільмів (All-Pairs)
py::tuple all_pairs_top_k(
    py::array_t<float, py::array::c_style | py::array::forcecast> embeddings,
    int k
) {
    if (embeddings.ndim() != 2) {
        throw std::runtime_error("Очікується двовимірний масив (N, D)");
    }

    auto mat = embeddings.unchecked<2>();
    const py::ssize_t n_items = mat.shape(0);
    const py::ssize_t dim = mat.shape(1);
    const int safe_k = std::min(std::max(k, 1), static_cast<int>(n_items) - 1);

    const std::vector<float> norms = compute_row_norms(mat);

    py::array_t<int> all_indices({n_items, static_cast<py::ssize_t>(safe_k)});
    py::array_t<float> all_scores({n_items, static_cast<py::ssize_t>(safe_k)});
    auto idx_out = all_indices.mutable_unchecked<2>();
    auto scr_out = all_scores.mutable_unchecked<2>();

    std::vector<std::pair<float, int>> row_scores(n_items - 1);

    for (py::ssize_t i = 0; i < n_items; ++i) {
        const float norm_i = norms[i];
        py::ssize_t pos = 0;

        for (py::ssize_t j = 0; j < n_items; ++j) {
            if (i == j) continue;

            float sim = 0.0f;
            if (norm_i > 1e-12f && norms[j] > 1e-12f) {
                float dot = 0.0f;
                for (py::ssize_t d = 0; d < dim; ++d) {
                    dot += mat(i, d) * mat(j, d);
                }
                sim = dot / (norm_i * norms[j]);
            }
            row_scores[pos++] = {sim, static_cast<int>(j)};
        }

        std::partial_sort(
            row_scores.begin(),
            row_scores.begin() + safe_k,
            row_scores.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            }
        );

        for (int rank = 0; rank < safe_k; ++rank) {
            idx_out(i, rank) = row_scores[rank].second;
            scr_out(i, rank) = row_scores[rank].first;
        }
    }

    return py::make_tuple(all_indices, all_scores);
}

PYBIND11_MODULE(fast_similarity, m) {
    m.doc() = "C++ extension for fast cosine similarity and Top-K retrieval";
    m.def(
        "top_k_similar_items",
        &top_k_similar_items,
        py::arg("embeddings"),
        py::arg("query_idx"),
        py::arg("k") = 10,
        "Знаходить Top-K найближчих за косинусною подібністю об'єктів для query_idx"
    );
    m.def(
        "all_pairs_top_k",
        &all_pairs_top_k,
        py::arg("embeddings"),
        py::arg("k") = 10,
        "Знаходить Top-K схожих об'єктів для кожного рядка матриці ембедінгів"
    );
}