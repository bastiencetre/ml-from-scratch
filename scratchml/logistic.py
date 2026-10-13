"""Binary logistic regression trained with Newton's method, plus classification metrics."""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    # Clipping avoids overflow in exp for very confident predictions.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Minimises the (optionally class-weighted) cross-entropy plus an L2 penalty.

    Newton's method (iteratively reweighted least squares) is used: the loss is convex with a
    cheap exact Hessian X' W X, so it converges in about ten iterations where gradient descent
    needs thousands. ``class_weight={1: 2.0}`` makes errors on class 1 twice as costly, which
    moves the decision boundary and trades false negatives for false positives.
    """

    def __init__(self, l2: float = 1e-6, class_weight: dict | None = None, max_iter: int = 100, tol: float = 1e-10):
        self.l2, self.class_weight, self.max_iter, self.tol = l2, class_weight, max_iter, tol

    def fit(self, X, y):
        X1 = np.column_stack([np.ones(len(X)), X])
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y))
        for label, weight in (self.class_weight or {}).items():
            w[y == label] = weight
        penalty = self.l2 * np.eye(X1.shape[1])
        penalty[0, 0] = 0.0                                   # the intercept is not penalised
        beta = np.zeros(X1.shape[1])
        for self.n_iter_ in range(1, self.max_iter + 1):
            p = sigmoid(X1 @ beta)
            gradient = X1.T @ (w * (p - y)) / len(y) + penalty @ beta
            hessian = (X1 * (w * p * (1 - p))[:, None]).T @ X1 / len(y) + penalty
            step = np.linalg.solve(hessian + 1e-12 * np.eye(len(beta)), gradient)
            beta -= step
            if np.abs(step).max() < self.tol:
                break
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes: int = 2) -> np.ndarray:
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def roc_curve(y_true, scores):
    """False and true positive rates for every threshold, from the sorted scores (O(n log n))."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")
    y_sorted, s_sorted = y_true[order], np.asarray(scores)[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    last_of_ties = np.r_[s_sorted[1:] != s_sorted[:-1], True]   # one point per distinct score
    tpr = np.r_[0.0, tp[last_of_ties] / max(tp[-1], 1)]
    fpr = np.r_[0.0, fp[last_of_ties] / max(fp[-1], 1)]
    return fpr, tpr


def auc(fpr, tpr) -> float:
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
