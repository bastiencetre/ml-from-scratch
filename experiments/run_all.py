"""Run the five experiments of the README and write their figures to figures/.

    python experiments/run_all.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scratchml import datasets, linear, logistic, neural, svm, unsupervised  # noqa: E402

FIGURES = ROOT / "figures"


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def decision_map(ax, model, X, y, title, score=None):
    """Shade the plane with the model's prediction and overlay the data."""
    x0, x1 = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y0, y1 = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    values = (score or model.predict)(grid).reshape(xx.shape)
    ax.contourf(xx, yy, values, levels=20 if score else 1, cmap="RdBu_r", alpha=0.35)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=12, edgecolors="k", linewidths=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# 1 ---------------------------------------------------------------------------------------

def overfitting_and_regularisation():
    """Stopping distance of a car: the truth is a degree-2 polynomial of the speed."""
    speed, dist = datasets.make_braking(20, (30, 110), seed=0)
    speed_test, dist_test = datasets.make_braking(200, (10, 130), seed=1)
    degrees = range(1, 11)

    def design(x, degree, scaler=None):
        P = linear.polynomial_features(x / 100.0, degree)
        scaler = scaler or linear.Standardizer().fit(P)
        return scaler.transform(P), scaler

    train_err, test_err = [], []
    for k in degrees:
        P, scaler = design(speed, k)
        model = linear.LinearRegression().fit(P, dist)
        train_err.append(linear.mse(dist, model.predict(P)))
        test_err.append(linear.mse(dist_test, model.predict(design(speed_test, k, scaler)[0])))

    # Degree 10 with regularisation; alpha chosen by 5-fold cross-validation.
    P10, scaler10 = design(speed, 10)
    P10_test = design(speed_test, 10, scaler10)[0]

    folds = np.random.default_rng(0).permutation(len(dist)) % 5

    def cross_validation_error(make_model):
        errors = []
        for f in range(5):
            m = make_model().fit(P10[folds != f], dist[folds != f])
            errors.append(linear.mse(dist[folds == f], m.predict(P10[folds == f])))
        return np.mean(errors)

    lasso_args = dict(max_iter=3000, tol=1e-5)
    alphas = np.logspace(-3, 2, 16)
    ridge_alpha = alphas[np.argmin([cross_validation_error(lambda a=a: linear.Ridge(a)) for a in alphas])]
    lasso_alpha = alphas[np.argmin([cross_validation_error(lambda a=a: linear.Lasso(a, **lasso_args)) for a in alphas])]
    ridge = linear.Ridge(ridge_alpha).fit(P10, dist)
    lasso = linear.Lasso(lasso_alpha, max_iter=50000).fit(P10, dist)
    ols = linear.LinearRegression().fit(P10, dist)

    grid = np.linspace(10, 130, 300)
    G = design(grid, 10, scaler10)[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(list(degrees), train_err, "o-", label="training error (20 points, 30-110 km/h)")
    ax.semilogy(list(degrees), test_err, "s-", label="test error (200 points, 10-130 km/h)")
    ax.set(xlabel="polynomial degree", ylabel="mean squared error", title="Least squares: more degrees, worse predictions")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax = axes[1]
    ax.scatter(speed_test, dist_test, s=6, c="lightgray", label="test data")
    ax.scatter(speed, dist, c="k", s=22, zorder=3, label="training data")
    ax.plot(grid, datasets.braking_distance(grid), "k--", lw=1, label="physics")
    for model, name in ((ols, "least squares"), (ridge, f"Ridge (alpha={ridge_alpha:.2g})"),
                        (lasso, f"Lasso (alpha={lasso_alpha:.2g})")):
        ax.plot(grid, model.predict(G), label=name)
    ax.set(ylim=(-20, 220), xlabel="speed (km/h)", ylabel="stopping distance (m)", title="Degree-10 models")
    ax.legend(fontsize=8)

    ax = axes[2]
    path = np.array([linear.Lasso(a, **lasso_args).fit(P10, dist).coef_ for a in alphas])
    for j in range(10):
        ax.semilogx(alphas, path[:, j], label=f"x^{j + 1}" if j < 3 else None)
    ax.axvline(lasso_alpha, color="gray", ls=":")
    ax.set(xlabel="alpha", ylabel="coefficient (standardised features)", title="Lasso path: coefficients vanish one by one")
    ax.legend(fontsize=8)
    save(fig, "1_regularisation.png")

    print("1. Regularisation (test MSE, degree 10):")
    for model, name in ((ols, "least squares"), (ridge, "ridge"), (lasso, "lasso")):
        print(f"   {name:<14} {linear.mse(dist_test, model.predict(P10_test)):12.1f}")
    print(f"   lasso keeps powers {(np.flatnonzero(lasso.coef_) + 1).tolist()} of the speed")
    print(f"   best degree for plain least squares: {int(np.argmin(test_err)) + 1}")


# 2 ---------------------------------------------------------------------------------------

def wine_classification():
    X, y, names = datasets.load_wine()
    two = [names.index("Ash"), names.index("Acl")]
    scaler = linear.Standardizer()
    X2 = scaler.fit_transform(X[:, two])
    model = logistic.LogisticRegression().fit(X2, y)
    weighted = logistic.LogisticRegression(class_weight={1: 4.0}).fit(X2, y)

    Xtr, Xte, ytr, yte = datasets.train_test_split(X, y, 0.3, seed=0)
    full_scaler = linear.Standardizer().fit(Xtr)
    full = logistic.LogisticRegression(l2=1e-2).fit(full_scaler.transform(Xtr), ytr)
    acc_full = logistic.accuracy(yte, full.predict(full_scaler.transform(Xte)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    decision_map(axes[0], model, X2, y, "Two features (ash, alkalinity): equal costs")
    decision_map(axes[1], weighted, X2, y, "Errors on cultivar 2 cost 4x more")
    fpr, tpr = logistic.roc_curve(y, model.predict_proba(X2))
    axes[2].plot(fpr, tpr, label=f"2 features, AUC = {logistic.auc(fpr, tpr):.3f}")
    fpr13, tpr13 = logistic.roc_curve(yte, full.predict_proba(full_scaler.transform(Xte)))
    axes[2].plot(fpr13, tpr13, label=f"13 features (test set), AUC = {logistic.auc(fpr13, tpr13):.3f}")
    axes[2].plot([0, 1], [0, 1], "k:", lw=1)
    axes[2].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves")
    axes[2].legend(fontsize=8)
    save(fig, "2_logistic_wine.png")

    print("2. Wine cultivars, logistic regression:")
    print(f"   2 features : accuracy {logistic.accuracy(y, model.predict(X2)):.3f}, Newton iterations {model.n_iter_}")
    print(f"   confusion  : {logistic.confusion_matrix(y, model.predict(X2)).tolist()}"
          f" -> weighted {logistic.confusion_matrix(y, weighted.predict(X2)).tolist()}")
    print(f"   13 features: test accuracy {acc_full:.3f}")


# 3 ---------------------------------------------------------------------------------------

def nonlinear_classifiers():
    X, y = datasets.make_moons(500, 0.15, seed=0)
    X_val, y_val = datasets.make_moons(2000, 0.15, seed=1)
    logit = logistic.LogisticRegression().fit(X, y)
    net = neural.MLPClassifier(hidden=(16,), epochs=2000).fit(X, y, validation=(X_val, y_val))
    rbf = svm.SVC(C=1.0, kernel=svm.rbf_kernel(2.0)).fit(X, y)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    results = {}
    for ax, model, name in zip(axes, (logit, net, rbf), ("logistic regression", "MLP, 16 tanh units", "SVM, RBF kernel")):
        results[name] = logistic.accuracy(y_val, model.predict(X_val))
        decision_map(ax, model, X, y, f"{name}: {100 * results[name]:.1f} % on validation")
    axes[2].scatter(*rbf.support_vectors_.T, s=40, facecolors="none", edgecolors="k", linewidths=1)
    axes[3].semilogy(net.train_loss_, label="training")
    axes[3].semilogy(net.val_loss_, label="validation")
    axes[3].set(xlabel="epoch", ylabel="cross-entropy", title="MLP learning curves")
    axes[3].legend()
    save(fig, "3_nonlinear_classifiers.png")

    widths = (1, 2, 4, 8, 16, 32)
    by_width = [logistic.accuracy(y_val, neural.MLPClassifier(hidden=(h,), epochs=1500).fit(X, y).predict(X_val))
                for h in widths]
    print("3. Two moons (validation accuracy):")
    for name, acc in results.items():
        print(f"   {name:<22} {acc:.3f}")
    print(f"   SVM support vectors: {len(rbf.support_)} of {len(y)}")
    print("   MLP by hidden width: " + ", ".join(f"{h}: {a:.3f}" for h, a in zip(widths, by_width)))


# 4 ---------------------------------------------------------------------------------------

def digits_pca():
    X, y = datasets.load_digits()
    X_train, y_train, X_test, y_test = X[:1500], y[:1500], X[1500:], y[1500:]
    pca = unsupervised.PCA(64).fit(X_train)

    # 3 versus 9: binary logistic regression on the first q principal components.
    tr, te = np.isin(y_train, (3, 9)), np.isin(y_test, (3, 9))
    qs = (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64)
    Z_tr, Z_te = pca.transform(X_train[tr]), pca.transform(X_test[te])
    train_acc, test_acc = [], []
    for q in qs:
        scaler = linear.Standardizer().fit(Z_tr[:, :q])
        clf = logistic.LogisticRegression(l2=1e-3).fit(scaler.transform(Z_tr[:, :q]), y_train[tr] == 9)
        train_acc.append(logistic.accuracy(y_train[tr] == 9, clf.predict(scaler.transform(Z_tr[:, :q]))))
        test_acc.append(logistic.accuracy(y_test[te] == 9, clf.predict(scaler.transform(Z_te[:, :q]))))

    fig = plt.figure(figsize=(16, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 12)
    for i in range(8):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        ax.imshow(pca.components_[i].reshape(28, 28), cmap="RdBu_r")
        ax.set_title(f"PC {i + 1}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(grid[:, 4:8])
    Z2 = pca.transform(X_train)[:, :2]
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=y_train, cmap="tab10", s=6)
    ax.legend(*sc.legend_elements(), title="digit", fontsize=6, ncol=2, loc="upper right")
    ax.set(title="1 500 digits on the first two components", xticks=[], yticks=[])
    ax = fig.add_subplot(grid[:, 8:])
    ax.plot(qs, train_acc, "o-", label="training")
    ax.plot(qs, test_acc, "s-", label="test")
    ax.set(xscale="log", xlabel="number of principal components q", ylabel="accuracy, 3 versus 9",
           title="Logistic regression after PCA")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    save(fig, "4_pca_digits.png")

    ratio = unsupervised.PCA(784).fit(X_train).explained_variance_ratio_.cumsum()
    best = int(np.argmax(test_acc))
    print("4. PCA on handwritten digits:")
    print(f"   components for 90 % of the variance: {int(np.searchsorted(ratio, 0.90)) + 1} of 784")
    print(f"   3 vs 9 test accuracy: q=2 -> {test_acc[1]:.3f}, best q={qs[best]} -> {test_acc[best]:.3f}, "
          f"q=64 -> {test_acc[-1]:.3f} (train {train_acc[-1]:.3f})")


# 5 ---------------------------------------------------------------------------------------

def colour_quantisation():
    from matplotlib import cbook

    image = plt.imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)).astype(float) / 255.0
    pixels = image.reshape(-1, 3)
    sample = pixels[np.random.default_rng(0).choice(len(pixels), 20000, replace=False)]
    ks = (2, 4, 8, 16)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(16, 4.2), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title(f"original ({len(np.unique((pixels * 255).astype(int), axis=0)):,} colours)", fontsize=10)
    print("5. k-means colour quantisation:")
    for ax, k in zip(axes[1:], ks):
        km = unsupervised.KMeans(k, n_init=2).fit(sample)
        quantised = km.cluster_centers_[km.predict(pixels)]
        error = float(np.sqrt(np.mean((quantised - pixels) ** 2)))
        ax.imshow(quantised.reshape(image.shape))
        ax.set_title(f"{k} colours ({int(np.log2(k))} bits/pixel), RMS error {error:.3f}", fontsize=10)
        print(f"   k={k:<2} RMS error {error:.3f}")
    for ax in axes:
        ax.axis("off")
    save(fig, "5_kmeans_colours.png")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    overfitting_and_regularisation()
    wine_classification()
    nonlinear_classifiers()
    digits_pca()
    colour_quantisation()
