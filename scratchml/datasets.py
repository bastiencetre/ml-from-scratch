"""Small datasets: two real ones shipped in data/, and synthetic generators."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load_wine(classes=(1, 2)):
    """UCI Wine: 13 chemical measurements of wines from three cultivars.

    Returns (X, y, feature_names) restricted to two cultivars, y in {0, 1}.
    """
    names = (DATA / "wine.csv").read_text().splitlines()[0].split(",")[1:]
    table = np.loadtxt(DATA / "wine.csv", delimiter=",", skiprows=1)
    keep = np.isin(table[:, 0], classes)
    return table[keep, 1:], (table[keep, 0] == classes[1]).astype(int), names


def load_digits():
    """3 000 handwritten digits from MNIST: images (n, 784) scaled to [0, 1], labels (n,)."""
    with np.load(DATA / "mnist_3000.npz") as f:
        return f["images"].astype(float) / 255.0, f["labels"].astype(int)


def make_moons(n=500, noise=0.1, seed=0):
    """Two interleaving half circles: not linearly separable."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t0, t1 = rng.uniform(0, np.pi, n0), rng.uniform(0, np.pi, n - n0)
    X = np.vstack([np.column_stack([np.cos(t0), np.sin(t0)]),
                   np.column_stack([1 - np.cos(t1), 0.5 - np.sin(t1)])])
    y = np.r_[np.zeros(n0, int), np.ones(n - n0, int)]
    return X + rng.normal(0, noise, X.shape), y


def make_circles(n=400, factor=0.3, noise=0.05, seed=0):
    """A small circle inside a large one."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t = rng.uniform(0, 2 * np.pi, n)
    radius = np.r_[np.ones(n0), factor * np.ones(n - n0)]
    X = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
    y = np.r_[np.zeros(n0, int), np.ones(n - n0, int)]
    return X + rng.normal(0, noise, X.shape), y


def make_braking(n=20, speed_range=(30, 110), reaction_time=1.0, deceleration=7.0, noise=5.0, seed=0):
    """Stopping distance (m) of a car against its speed (km/h).

    Physics: distance = v t_r + v^2 / (2 a) with v in m/s, so the truth is a degree-2
    polynomial of the speed. Returns (speed, distance).
    """
    rng = np.random.default_rng(seed)
    speed = np.sort(rng.uniform(*speed_range, n))
    return speed, braking_distance(speed, reaction_time, deceleration) + rng.normal(0, noise, n)


def braking_distance(speed_kmh, reaction_time=1.0, deceleration=7.0):
    v = np.asarray(speed_kmh) / 3.6
    return v * reaction_time + v**2 / (2 * deceleration)


def train_test_split(X, y, test_size=0.3, seed=0):
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(round(test_size * len(y)))
    test, train = order[:n_test], order[n_test:]
    return X[train], X[test], y[train], y[test]
