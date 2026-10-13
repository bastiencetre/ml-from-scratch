"""Linear models for regression: least squares, Ridge and Lasso, with polynomial features."""

from __future__ import annotations

import numpy as np


def polynomial_features(x, degree: int) -> np.ndarray:
    """Columns x, x^2, ..., x^degree of a 1-D input (no constant column: models fit an intercept)."""
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x**d for d in range(1, degree + 1)])


class Standardizer:
    """Centre and scale features. Regularised models need it: the penalty treats every
    coefficient alike, so features must live on the same scale."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class _LinearModel:
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def _center(self, X, y):
        self._x_mean, self._y_mean = X.mean(axis=0), y.mean()
        return X - self._x_mean, y - self._y_mean

    def _set(self, coef):
        self.coef_ = coef
        self.intercept_ = self._y_mean - self._x_mean @ coef
        return self


class LinearRegression(_LinearModel):
    """Ordinary least squares, solved with a QR/SVD-based solver rather than the normal
    equations, whose condition number is the square of that of X."""

    def fit(self, X, y):
        Xc, yc = self._center(X, y)
        return self._set(np.linalg.lstsq(Xc, yc, rcond=None)[0])


class Ridge(_LinearModel):
    """min ||y - X b||^2 + alpha ||b||^2, closed form b = (X'X + alpha I)^-1 X'y."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xc, yc = self._center(X, y)
        gram = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        return self._set(np.linalg.solve(gram, Xc.T @ yc))


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


class Lasso(_LinearModel):
    """min 1/(2n) ||y - X b||^2 + alpha ||b||_1 by cyclic coordinate descent.

    Each coordinate has a closed-form update (a soft-thresholding), and the residual is updated
    incrementally so one sweep costs O(n p). The L1 penalty sets coefficients exactly to zero,
    which performs variable selection.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 5000, tol: float = 1e-8):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X, y):
        Xc, yc = self._center(X, y)
        n, p = Xc.shape
        coef = np.zeros(p)
        residual = yc.copy()
        col_norm2 = (Xc**2).sum(axis=0) / n
        for self.n_iter_ in range(1, self.max_iter + 1):
            largest_change = 0.0
            for j in range(p):
                if col_norm2[j] == 0:
                    continue
                rho = Xc[:, j] @ residual / n + col_norm2[j] * coef[j]
                new = soft_threshold(rho, self.alpha) / col_norm2[j]
                if new != coef[j]:
                    residual -= Xc[:, j] * (new - coef[j])
                    largest_change = max(largest_change, abs(new - coef[j]))
                    coef[j] = new
            if largest_change < self.tol:
                break
        return self._set(coef)


def mse(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
