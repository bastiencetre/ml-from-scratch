"""Unsupervised learning: principal component analysis and k-means."""

from __future__ import annotations

import numpy as np


class PCA:
    """Projection on the directions of largest variance.

    Computed from the SVD of the centred data, which is numerically more stable than
    diagonalising the covariance matrix and gives the same directions.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        variance = s**2 / (len(X) - 1)
        self.explained_variance_ratio_ = variance[: self.n_components] / variance.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KMeans:
    """Lloyd's algorithm with k-means++ seeding.

    k-means++ picks each new centre with probability proportional to its squared distance to
    the centres already chosen: spread-out seeds avoid most bad local minima, so a few restarts
    are enough.
    """

    def __init__(self, n_clusters: int, n_init: int = 3, max_iter: int = 100, tol: float = 1e-6, seed: int = 0):
        self.n_clusters, self.n_init, self.max_iter, self.tol, self.seed = n_clusters, n_init, max_iter, tol, seed

    @staticmethod
    def _sq_distances(X, centers):
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, without building an (n, k, d) array.
        return (X**2).sum(axis=1)[:, None] - 2 * X @ centers.T + (centers**2).sum(axis=1)[None]

    def _seed(self, X, rng):
        centers = [X[rng.integers(len(X))]]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)
        for _ in range(1, self.n_clusters):
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
            d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
        return np.array(centers)

    def _run(self, X, rng):
        centers = self._seed(X, rng)
        for _ in range(self.max_iter):
            labels = self._sq_distances(X, centers).argmin(axis=1)
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, X)
            counts = np.bincount(labels, minlength=self.n_clusters)
            new = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers)
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.tol:
                break
        d2 = self._sq_distances(X, centers)
        labels = d2.argmin(axis=1)
        return centers, labels, float(d2[np.arange(len(X)), labels].sum())

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        best = min((self._run(X, rng) for _ in range(self.n_init)), key=lambda r: r[2])
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        return self._sq_distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)
