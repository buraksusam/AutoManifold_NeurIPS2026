import numpy as np
import pytest
from pipeline.engines.metrics_engine import QuantitativeAnalystEngine
from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def engine():
    return QuantitativeAnalystEngine()


def _blobs(n=120, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [10, 10], [0, 10]])
    labels = rng.randint(0, 3, n)
    pts = centers[labels] + rng.randn(n, 2) * 0.5
    return pts.astype(float), labels


def test_rnx_auc_identity_is_high(engine):
    X, _ = _blobs()
    assert engine.rnx_auc(X, X.copy()) > 0.95


def test_rnx_auc_random_is_lower_than_identity(engine):
    X, _ = _blobs()
    rng = np.random.RandomState(1)
    Y = rng.randn(*X.shape)
    assert engine.rnx_auc(X, Y) < engine.rnx_auc(X, X.copy())


def test_shepard_pearson_identity_is_one(engine):
    X, _ = _blobs()
    d = squareform(pdist(X))
    assert engine.shepard_pearson(d, d.copy()) == pytest.approx(1.0, abs=1e-6)


def test_knn_accuracy_separable_is_high(engine):
    X, labels = _blobs()
    assert engine.knn_accuracy(X, labels, k=10) > 0.9


def test_kmeans_nmi_separable_is_high(engine):
    X, labels = _blobs()
    assert engine.kmeans_nmi(X, labels) > 0.8


def test_knn_accuracy_shuffled_labels_is_low(engine):
    X, labels = _blobs()
    rng = np.random.RandomState(2)
    shuffled = rng.permutation(labels)
    assert engine.knn_accuracy(X, shuffled, k=10) < 0.6
