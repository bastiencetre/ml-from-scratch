"""A multilayer perceptron for binary classification, with backpropagation written by hand."""

from __future__ import annotations

import numpy as np

from .logistic import sigmoid

ACTIVATIONS = {
    # name: (function, derivative expressed with the activation value a)
    "tanh": (np.tanh, lambda a: 1.0 - a**2),
    "sigmoid": (sigmoid, lambda a: a * (1.0 - a)),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(float)),
}


class MLPClassifier:
    """Fully connected network: hidden layers with the chosen activation, sigmoid output,
    cross-entropy loss, full-batch gradient descent with momentum.

    Backpropagation is the chain rule applied layer by layer from the output. With a sigmoid
    output and the cross-entropy, the error on the last pre-activation is simply
    (prediction - label), then  delta_l = (delta_{l+1} W_{l+1}') * f'(a_l).
    """

    def __init__(self, hidden=(16,), activation="tanh", lr=0.5, momentum=0.9, epochs=2000, seed=0):
        self.hidden, self.activation = tuple(hidden), activation
        self.lr, self.momentum, self.epochs, self.seed = lr, momentum, epochs, seed

    def _init(self, n_features):
        rng = np.random.default_rng(self.seed)
        sizes = (n_features, *self.hidden, 1)
        # Glorot initialisation keeps the variance of the signal stable across layers.
        self.weights_ = [rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.biases_ = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X):
        f, _ = ACTIVATIONS[self.activation]
        activations = [X]
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            activations.append(f(activations[-1] @ W + b))
        output = sigmoid(activations[-1] @ self.weights_[-1] + self.biases_[-1]).ravel()
        return activations, output

    def _gradients(self, X, y):
        _, df = ACTIVATIONS[self.activation]
        activations, output = self._forward(X)
        delta = ((output - y) / len(y))[:, None]
        grads_w, grads_b = [], []
        for layer in range(len(self.weights_) - 1, -1, -1):
            grads_w.append(activations[layer].T @ delta)
            grads_b.append(delta.sum(axis=0))
            if layer > 0:
                delta = (delta @ self.weights_[layer].T) * df(activations[layer])
        return grads_w[::-1], grads_b[::-1]

    @staticmethod
    def cross_entropy(y, p) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y, validation=None):
        """Train; if ``validation=(X_val, y_val)`` is given, its loss is recorded at each epoch."""
        y = np.asarray(y, dtype=float)
        self._init(X.shape[1])
        vel_w = [np.zeros_like(W) for W in self.weights_]
        vel_b = [np.zeros_like(b) for b in self.biases_]
        self.train_loss_, self.val_loss_ = [], []
        for _ in range(self.epochs):
            grads_w, grads_b = self._gradients(X, y)
            for i in range(len(self.weights_)):
                vel_w[i] = self.momentum * vel_w[i] - self.lr * grads_w[i]
                vel_b[i] = self.momentum * vel_b[i] - self.lr * grads_b[i]
                self.weights_[i] += vel_w[i]
                self.biases_[i] += vel_b[i]
            self.train_loss_.append(self.cross_entropy(y, self.predict_proba(X)))
            if validation is not None:
                self.val_loss_.append(self.cross_entropy(validation[1], self.predict_proba(validation[0])))
        return self

    def predict_proba(self, X):
        return self._forward(X)[1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
