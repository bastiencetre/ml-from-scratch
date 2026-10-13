"""Kernel support vector machine trained with a simplified SMO algorithm."""

from __future__ import annotations

import numpy as np


def linear_kernel(A, B):
    return A @ B.T


def rbf_kernel(gamma: float):
    def kernel(A, B):
        d2 = (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None]
        return np.exp(-gamma * np.maximum(d2, 0.0))
    return kernel


class SVC:
    """Soft-margin SVM, dual problem:

        max  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i, x_j)
        s.t. 0 <= a_i <= C,  sum_i a_i y_i = 0.

    Sequential minimal optimisation solves it two multipliers at a time, because a pair is the
    smallest set that can move while keeping sum a_i y_i = 0, and the two-variable problem has
    a closed-form solution. Pairs are chosen with the maximal-violating-pair rule, and training
    stops when the KKT conditions hold up to ``tol``.
    Only the points with a_i > 0 (the support vectors) appear in the decision function.
    """

    def __init__(self, C: float = 1.0, kernel=linear_kernel, tol: float = 1e-3, max_iter: int = 100_000):
        self.C, self.kernel, self.tol, self.max_iter = C, kernel, tol, max_iter

    def fit(self, X, y):
        y = np.where(np.asarray(y) > 0, 1.0, -1.0)
        n = len(y)
        K = self.kernel(X, X)
        alpha = np.zeros(n)
        grad = -np.ones(n)                    # gradient of the dual objective (to minimise)
        for self.n_iter_ in range(1, self.max_iter + 1):
            # i can still increase y_i a_i, j can still decrease it.
            up = ((y > 0) & (alpha < self.C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < self.C))
            score = -y * grad
            i = np.flatnonzero(up)[np.argmax(score[up])]
            j = np.flatnonzero(low)[np.argmin(score[low])]
            if score[i] - score[j] < self.tol:
                break
            eta = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
            step = (score[i] - score[j]) / eta
            # Keep both multipliers inside [0, C].
            max_i = self.C - alpha[i] if y[i] > 0 else alpha[i]
            max_j = alpha[j] if y[j] > 0 else self.C - alpha[j]
            step = min(step, max_i, max_j)
            alpha[i] += y[i] * step
            alpha[j] -= y[j] * step
            grad += step * y * (K[:, i] - K[:, j])
        support = alpha > 1e-8
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.dual_coef_ = (alpha * y)[support]
        free = support & (alpha < self.C - 1e-8)
        margin_points = free if free.any() else support
        self.intercept_ = float(np.mean(y[margin_points] - K[margin_points][:, support] @ self.dual_coef_))
        return self

    def decision_function(self, X):
        return self.kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
