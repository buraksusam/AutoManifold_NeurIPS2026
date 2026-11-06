"""Single evaluator that every experiment arm routes its final embedding through.

Computes the 6 core metrics + 4 held-out metrics identically for t-SNE
(default/tuned/single-metric) and AutoManifold so numbers are directly
comparable. Held-out metrics are evaluation-only and never enter any objective.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pipeline.engines.metrics_engine import QuantitativeAnalystEngine

_CORE_KEYS = ("Trustworthiness_AUC", "Continuity_AUC", "Spearman",
              "Silhouette", "Stress", "LOF_Median")


def _first(block: dict, prefix: str):
    for key, val in block.items():
        if key.startswith(prefix):
            return val
    return None


def precompute_hd(X) -> dict:
    """Precompute the HD-side artifacts that depend ONLY on X, so many
    embeddings of the same dataset can be evaluated without recomputing pairwise
    distances / argsorts each time. Pass the returned dict as ``**hd_cache`` to
    ``evaluate``.

    The values are exactly what ``evaluate`` / ``compute_all`` compute
    internally when they are not supplied (``hd_argsort = hd_dists.argsort(1)``,
    ``hd_ranks = hd_argsort.argsort(1)``), so results are bit-for-bit identical.
    """
    X = np.asarray(X, dtype=float)
    hd_dists = squareform(pdist(X))
    hd_argsort = hd_dists.argsort(axis=1)
    hd_ranks = hd_argsort.argsort(axis=1)
    return {"hd_dists": hd_dists, "hd_argsort": hd_argsort, "hd_ranks": hd_ranks}


def evaluate(X, embedding, labels=None, *, k=10, seed=0,
             hd_dists=None, hd_argsort=None, hd_ranks=None) -> dict:
    eng = QuantitativeAnalystEngine()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(embedding, dtype=float)
    lab = None if labels is None else np.asarray(labels)

    # HD-side artifacts depend only on X. Reuse them across embeddings when
    # supplied (see precompute_hd); otherwise compute as before.
    if hd_dists is None:
        hd_dists = squareform(pdist(X))
    snap = eng.compute_all(X, Y, labels=lab, k=k,
                           hd_dists=hd_dists, hd_argsort=hd_argsort, hd_ranks=hd_ranks)
    g = snap.get("Global Structure & HD<->2D Correlation", {})
    n_block = snap.get("Neighborhood Preservation", {})
    cl = snap.get("Clustering & Label Agreement", {})
    lof = snap.get("Outlier Detection", {})

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    ld_dists = squareform(pdist(Y))

    out = {
        "Trustworthiness_AUC": _num(n_block.get("Trustworthiness AUC")),
        "Continuity_AUC": _num(n_block.get("Continuity AUC")),
        "Spearman": _num(g.get("Spearman Correlation")),
        "Silhouette": _num(cl.get("Silhouette Score")),
        "Stress": _num(g.get("Stress")),
        "LOF_Median": _num(_first(lof, "LOF Median")),
        "RNX_AUC": _num(eng.rnx_auc(X, Y)),
        "Shepard_Pearson": _num(eng.shepard_pearson(hd_dists, ld_dists)),
        "KNN_Accuracy": _num(eng.knn_accuracy(Y, lab, k=k)) if lab is not None else float("nan"),
        "KMeans_NMI": _num(eng.kmeans_nmi(Y, lab, seed=seed)) if lab is not None else float("nan"),
        "_meta": {"n": int(X.shape[0]), "k": k, "has_labels": lab is not None},
    }
    return out
