import math
import numpy as np
import pytest
from pipeline.engines.unified_eval import evaluate, precompute_hd

CORE = ["Trustworthiness_AUC", "Continuity_AUC", "Spearman", "Silhouette", "Stress", "LOF_Median"]
HELDOUT = ["RNX_AUC", "Shepard_Pearson", "KNN_Accuracy", "KMeans_NMI"]


def _blobs(n=120, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [10, 10], [0, 10]])
    labels = rng.randint(0, 3, n)
    X = np.hstack([centers[labels] + rng.randn(n, 2) * 0.5, rng.randn(n, 6)])
    return X.astype(float), labels


def test_evaluate_returns_all_ten_keys():
    X, labels = _blobs()
    out = evaluate(X, X[:, :2], labels)
    for key in CORE + HELDOUT:
        assert key in out, f"missing {key}"


def test_supervised_metrics_are_nan_without_labels():
    X, _ = _blobs()
    out = evaluate(X, X[:, :2], labels=None)
    assert math.isnan(out["KNN_Accuracy"])
    assert math.isnan(out["KMeans_NMI"])
    # Unsupervised held-outs still computed.
    assert not math.isnan(out["RNX_AUC"])
    assert not math.isnan(out["Shepard_Pearson"])


def test_precompute_hd_cache_is_identical_to_uncached():
    # The HD cache is an optimization only: passing precompute_hd(X) must give
    # bit-for-bit the same metrics as computing them inline.
    X, labels = _blobs()
    Y = X[:, :2] + 1e-3 * np.random.RandomState(1).randn(*X[:, :2].shape)
    base = evaluate(X, Y, labels)
    cached = evaluate(X, Y, labels, **precompute_hd(X))
    for key in CORE + HELDOUT:
        a, b = base[key], cached[key]
        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), key
        else:
            assert a == pytest.approx(b), key


def test_meta_block_present():
    X, labels = _blobs()
    out = evaluate(X, X[:, :2], labels)
    assert out["_meta"]["n"] == X.shape[0]
    assert out["_meta"]["has_labels"] is True
