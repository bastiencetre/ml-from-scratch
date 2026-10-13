"""Every model is checked against its scikit-learn counterpart on the same data."""

import numpy as np
import pytest
import sklearn.cluster
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.svm

from scratchml import datasets, linear, logistic, neural, svm, unsupervised


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = X @ np.array([3.0, 0.0, -2.0, 0.0, 0.0, 1.0]) + 0.5 + rng.normal(0, 0.3, 80)
    return X, y


def test_linear_regression(regression_data):
    X, y = regression_data
    ours = linear.LinearRegression().fit(X, y)
    ref = sklearn.linear_model.LinearRegression().fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_) and np.isclose(ours.intercept_, ref.intercept_)


@pytest.mark.parametrize("alpha", [0.1, 10.0])
def test_ridge(regression_data, alpha):
    X, y = regression_data
    ours = linear.Ridge(alpha).fit(X, y)
    ref = sklearn.linear_model.Ridge(alpha=alpha).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_) and np.isclose(ours.intercept_, ref.intercept_)


@pytest.mark.parametrize("alpha", [0.01, 0.3])
def test_lasso_and_its_sparsity(regression_data, alpha):
    X, y = regression_data
    ours = linear.Lasso(alpha).fit(X, y)
    ref = sklearn.linear_model.Lasso(alpha=alpha, tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-5)
    if alpha == 0.3:
        assert np.count_nonzero(ours.coef_) == 3          # the three true features survive


def test_logistic_regression_and_metrics():
    X, y, _ = datasets.load_wine()
    Xs = linear.Standardizer().fit_transform(X[:, [2, 3]])     # ash, alcalinity: classes overlap
    ours = logistic.LogisticRegression(l2=0.0).fit(Xs, y)
    ref = sklearn.linear_model.LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(Xs, y)
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-4)
    assert np.isclose(ours.intercept_, ref.intercept_[0], atol=1e-4)

    scores = ours.predict_proba(Xs)
    assert np.array_equal(logistic.confusion_matrix(y, ours.predict(Xs)),
                          sklearn.metrics.confusion_matrix(y, ref.predict(Xs)))
    fpr, tpr = logistic.roc_curve(y, scores)
    assert np.isclose(logistic.auc(fpr, tpr), sklearn.metrics.roc_auc_score(y, scores))


def test_class_weight_reduces_errors_on_the_heavier_class():
    X, y, _ = datasets.load_wine()
    Xs = linear.Standardizer().fit_transform(X[:, [2, 3]])
    plain = logistic.confusion_matrix(y, logistic.LogisticRegression().fit(Xs, y).predict(Xs))
    weighted = logistic.confusion_matrix(y, logistic.LogisticRegression(class_weight={1: 5.0}).fit(Xs, y).predict(Xs))
    assert weighted[1, 0] < plain[1, 0]                    # fewer missed class-1 samples


def test_pca():
    X, _ = datasets.load_digits()
    X = X[:500]
    ours = unsupervised.PCA(10).fit(X)
    ref = sklearn.decomposition.PCA(10, svd_solver="full").fit(X)
    assert np.allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_)
    assert np.allclose(np.abs(ours.components_ @ ref.components_.T), np.eye(10), atol=1e-6)   # same axes up to sign
    assert np.allclose(ours.inverse_transform(ours.transform(X)), ref.inverse_transform(ref.transform(X)))


def test_kmeans_reaches_the_same_inertia():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.4, (100, 2)) for c in ((0, 0), (4, 0), (0, 4), (4, 4))])
    ours = unsupervised.KMeans(4, n_init=5).fit(X)
    ref = sklearn.cluster.KMeans(4, n_init=10, random_state=0).fit(X)
    assert np.isclose(ours.inertia_, ref.inertia_, rtol=1e-6)
    assert np.array_equal(ours.predict(X), ours.labels_)


def test_backpropagation_matches_finite_differences():
    X, y = datasets.make_moons(40, seed=3)
    net = neural.MLPClassifier(hidden=(5, 4), activation="tanh")
    net._init(2)
    grads_w, grads_b = net._gradients(X, y.astype(float))
    eps = 1e-6
    for layer in range(3):
        W = net.weights_[layer]
        for index in [(0, 0), (W.shape[0] - 1, W.shape[1] - 1)]:
            old = W[index]
            W[index] = old + eps
            up = net.cross_entropy(y, net.predict_proba(X))
            W[index] = old - eps
            down = net.cross_entropy(y, net.predict_proba(X))
            W[index] = old
            assert np.isclose(grads_w[layer][index], (up - down) / (2 * eps), rtol=1e-4, atol=1e-8)


def test_mlp_separates_the_moons_where_logistic_regression_cannot():
    X, y = datasets.make_moons(400, noise=0.1, seed=0)
    X_test, y_test = datasets.make_moons(1000, noise=0.1, seed=1)
    net = neural.MLPClassifier(hidden=(16,), epochs=1500).fit(X, y)
    linear_model = logistic.LogisticRegression().fit(X, y)
    assert logistic.accuracy(y_test, net.predict(X_test)) > 0.98
    assert logistic.accuracy(y_test, linear_model.predict(X_test)) < 0.92


@pytest.mark.parametrize("kernel_name", ["linear", "rbf"])
def test_svm_matches_libsvm(kernel_name):
    X, y = datasets.make_circles(200, noise=0.1, seed=0) if kernel_name == "rbf" else datasets.make_moons(200, 0.2, 0)
    kernel = svm.rbf_kernel(2.0) if kernel_name == "rbf" else svm.linear_kernel
    ours = svm.SVC(C=1.0, kernel=kernel, tol=1e-5).fit(X, y)
    ref = sklearn.svm.SVC(C=1.0, kernel=kernel_name, gamma=2.0, tol=1e-5).fit(X, y)
    grid = np.random.default_rng(0).uniform(-1.5, 2.5, (300, 2))
    assert np.allclose(ours.decision_function(grid), ref.decision_function(grid), atol=1e-3)
    assert set(ours.support_) == set(ref.support_)
