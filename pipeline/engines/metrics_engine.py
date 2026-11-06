"""
Quantitative Analyst Engine

Computes all reliability metrics defined in the paper:
  Global:  Spearman ρs (Eq.1), Stress S (Eq.2), Mean Distance Ratio R (Eq.3)
  Local:   Trustworthiness T(k) (Eq.4), Continuity C(k) (Eq.5)
  Cluster: Silhouette Score
  Outlier: LOF statistics

Corresponds to the "Quantitative Analyst Engine" block in the paper's Figure 1.
"""

import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

logger = logging.getLogger(__name__)


class QuantitativeAnalystEngine:
    """Compute embedding reliability metrics."""

    # ------------------------------------------------------------------
    # Master method — compute everything at once
    # ------------------------------------------------------------------

    # Default k-sweep used for the AUC of T(k) / C(k). Logarithmic grid that
    # spans local→meso scales; values larger than n//2 are dropped so the
    # sweep is well-defined for small datasets too.
    DEFAULT_K_SWEEP: tuple = (5, 10, 20, 50, 100, 200)

    def compute_all(
        self,
        hd_data: np.ndarray,
        ld_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        k: int = 10,
        lof_neighbors: int = 20,
        k_sweep: Optional[tuple] = None,
        hd_dists: Optional[np.ndarray] = None,
        hd_argsort: Optional[np.ndarray] = None,
        hd_ranks: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute the full suite of metrics used in the paper.

        ``k_sweep`` controls the grid used for the T(k)/C(k) AUC. Pass
        ``False`` (or an empty iterable) to skip the sweep. Defaults to
        :pyattr:`DEFAULT_K_SWEEP`, clipped to k < n // 2 so it is well-defined
        on small datasets.

        ``hd_dists``, ``hd_argsort``, ``hd_ranks`` may be supplied to skip the
        HD-side recomputation (they only depend on ``hd_data`` and are stable
        across sweep cells on the same dataset).

        Returns:
            Dict matching the paper's multi-modal input schema under "metrics".
        """
        if hd_dists is None:
            logger.info("Computing HD pairwise distances …")
            hd_dists = squareform(pdist(hd_data))
        ld_dists = squareform(pdist(ld_data))

        logger.info("Computing global metrics …")
        spearman = self.spearman_distance_score(hd_dists, ld_dists)
        stress = self.stress_score(hd_dists, ld_dists)
        ratio = self.mean_distance_ratio(hd_dists, ld_dists)

        logger.info("Computing local metrics …")
        trust = self.trustworthiness(hd_dists, ld_dists, k, hd_ranks=hd_ranks)
        cont = self.continuity(hd_dists, ld_dists, k, hd_argsort=hd_argsort)

        # k-sweep + AUC — computed on the same precomputed distance matrices
        # so this only adds an O(n²) argsort and O(n · k_max) penalty pass.
        sweep_block = None
        if k_sweep is not False:
            sweep_grid = self._resolve_k_sweep(
                k_sweep if k_sweep is not None else self.DEFAULT_K_SWEEP,
                n=hd_dists.shape[0],
            )
            if sweep_grid:
                logger.info("Computing T/C k-sweep over %s …", sweep_grid)
                sweep_block = self.trustworthiness_continuity_sweep(
                    hd_dists, ld_dists, sweep_grid,
                    hd_argsort=hd_argsort, hd_ranks=hd_ranks,
                )

        logger.info("Computing cluster metrics …")
        sil = self.silhouette(ld_data, labels) if labels is not None else None

        logger.info("Computing LOF …")
        lof_stats = self.lof_statistics(ld_data, lof_neighbors)

        # Per-label centroids in 2D
        centroids = {}
        if labels is not None:
            for lab in np.unique(labels):
                mask = labels == lab
                centroids[str(lab)] = ld_data[mask].mean(axis=0).tolist()

        neighborhood_block: dict = {
            f"Trustworthiness (k={k})": round(trust, 4),
            f"Continuity (k={k})": round(cont, 4),
        }
        if sweep_block is not None:
            neighborhood_block["Trustworthiness AUC"] = round(
                sweep_block["trustworthiness_auc"], 4
            )
            neighborhood_block["Continuity AUC"] = round(
                sweep_block["continuity_auc"], 4
            )
            neighborhood_block["k_sweep"] = sweep_block["k_values"]
            neighborhood_block["Trustworthiness curve"] = [
                round(v, 4) for v in sweep_block["trustworthiness_curve"]
            ]
            neighborhood_block["Continuity curve"] = [
                round(v, 4) for v in sweep_block["continuity_curve"]
            ]

        return {
            "Global Structure & HD<->2D Correlation": {
                "Spearman Correlation": round(spearman, 4),
                "Stress": round(stress, 4),
                "Mean Distance Ratio": round(ratio, 4),
            },
            "Neighborhood Preservation": neighborhood_block,
            "Clustering & Label Agreement": {
                "Silhouette Score": round(sil, 4) if sil is not None else "N/A",
            },
            "Outlier Detection": lof_stats,
            "Cluster Centroids (2D)": centroids,
        }

    @staticmethod
    def _resolve_k_sweep(k_sweep, n: int) -> list:
        """Drop k values that are too large for ``n`` (need 2k < n - 1 so the
        normalizer is positive). Always keep at least k=5 if any k is valid."""
        if k_sweep is False or k_sweep is None:
            return []
        try:
            grid = sorted({int(k) for k in k_sweep if int(k) >= 1})
        except (TypeError, ValueError):
            return []
        # Keep k strictly less than n/2 so n*k*(2n - 3k - 1) > 0 with margin.
        valid = [k for k in grid if 2 * k < n - 1]
        return valid

    @staticmethod
    def trustworthiness_continuity_sweep(
        hd_dists: np.ndarray,
        ld_dists: np.ndarray,
        k_values,
        hd_argsort: Optional[np.ndarray] = None,
        hd_ranks: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute T(k) and C(k) at every k in ``k_values`` and the trapezoidal
        AUC of each curve, normalized so AUC ∈ [0, 1] independently of the grid.

        Reuses precomputed pairwise-distance argsorts so the sweep cost is
        O(n² log n) for the sorts plus O(n · k_max) for the penalty pass.

        Returns:
            dict with keys ``k_values``, ``trustworthiness_curve``,
            ``continuity_curve``, ``trustworthiness_auc``, ``continuity_auc``.
        """
        ks = sorted({int(k) for k in k_values if int(k) >= 1})
        if not ks:
            return {
                "k_values": [],
                "trustworthiness_curve": [],
                "continuity_curve": [],
                "trustworthiness_auc": float("nan"),
                "continuity_auc": float("nan"),
            }
        n = hd_dists.shape[0]
        k_max = ks[-1]

        # 0-indexed ranks: rank 0 = self (distance 0 to itself), rank 1..n-1 =
        # ascending distance. Used for the penalty (rank − k). hd_argsort /
        # hd_ranks may be passed in pre-computed (HD-side cache).
        if hd_argsort is None:
            hd_argsort = hd_dists.argsort(axis=1)
        if hd_ranks is None:
            hd_ranks = hd_argsort.argsort(axis=1)
        ld_argsort = ld_dists.argsort(axis=1)
        ld_ranks = ld_argsort.argsort(axis=1)

        # For each i, the HD ranks of its (1..k_max) LD-nearest neighbors.
        # Column p (0-indexed) = HD rank of the (p+1)-th closest LD neighbor.
        hd_ranks_in_ld_order = np.take_along_axis(
            hd_ranks, ld_argsort[:, 1 : k_max + 1], axis=1
        )
        ld_ranks_in_hd_order = np.take_along_axis(
            ld_ranks, hd_argsort[:, 1 : k_max + 1], axis=1
        )

        t_curve = []
        c_curve = []
        for k in ks:
            normalizer = n * k * (2 * n - 3 * k - 1)
            if normalizer <= 0:
                t_curve.append(float("nan"))
                c_curve.append(float("nan"))
                continue
            # Trustworthiness: penalize LD-NN whose HD rank > k.
            t_block = hd_ranks_in_ld_order[:, :k]
            t_pen = np.maximum(0, t_block - k).sum()
            t_curve.append(float(1.0 - (2.0 / normalizer) * t_pen))
            # Continuity: penalize HD-NN whose LD rank > k.
            c_block = ld_ranks_in_hd_order[:, :k]
            c_pen = np.maximum(0, c_block - k).sum()
            c_curve.append(float(1.0 - (2.0 / normalizer) * c_pen))

        # Trapezoidal AUC over the k-grid, normalized to [0, 1] regardless of
        # spacing. With a single k, fall back to the value itself.
        if len(ks) == 1:
            t_auc = t_curve[0]
            c_auc = c_curve[0]
        else:
            xs = np.asarray(ks, dtype=float)
            span = xs[-1] - xs[0]
            t_auc = float(np.trapz(np.asarray(t_curve), xs) / span)
            c_auc = float(np.trapz(np.asarray(c_curve), xs) / span)

        return {
            "k_values": ks,
            "trustworthiness_curve": t_curve,
            "continuity_curve": c_curve,
            "trustworthiness_auc": t_auc,
            "continuity_auc": c_auc,
        }

    # ------------------------------------------------------------------
    # Global Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def spearman_distance_score(
        hd_dists: np.ndarray, ld_dists: np.ndarray
    ) -> float:
        """
        Spearman rank correlation of pairwise distances (paper Eq. 1).
        Higher (→ 1) = better preservation of global distance ranking.
        """
        idx = np.triu_indices_from(hd_dists, k=1)
        rho, _ = spearmanr(hd_dists[idx], ld_dists[idx])
        return float(rho)

    @staticmethod
    def stress_score(hd_dists: np.ndarray, ld_dists: np.ndarray) -> float:
        """
        Kruskal's Stress-1, with the textbook optimal scalar rescaling:

            σ_1 = sqrt( min_a Σ(a·d_ij - δ_ij)²  /  Σ δ_ij² )
                  with optimal a* = Σ(d_ij·δ_ij) / Σ d_ij²

        where δ_ij are HD distances and d_ij are LD distances. The scalar
        a* makes the metric invariant to the absolute scale of the LD
        embedding, which is the standard formulation used for cross-method
        DR comparison (Kruskal 1964; Borg & Groenen 2005).

        Every iteration's LD embedding is persisted as ``iter_NNN.npy``
        (plus ``best_embedding.npy``), so Stress and any other metric can be
        recomputed in place without re-executing the synthesised algorithm.

        Lower (→ 0) = better geometry preservation.
        """
        iu = np.triu_indices_from(hd_dists, k=1)
        delta = hd_dists[iu]
        d = ld_dists[iu]
        denom_d2 = float(np.dot(d, d))
        if denom_d2 == 0:
            return 0.0
        a_opt = float(np.dot(d, delta) / denom_d2)
        num = float(np.sum((a_opt * d - delta) ** 2))
        den = float(np.sum(delta ** 2))
        if den == 0:
            return 0.0
        return float(np.sqrt(num / den))

    @staticmethod
    def mean_distance_ratio(hd_dists: np.ndarray, ld_dists: np.ndarray) -> float:
        """
        Mean pairwise distance ratio (paper Eq. 3).
        R ≈ 1 indicates low global scaling distortion.
        """
        idx = np.triu_indices_from(hd_dists, k=1)
        hd = hd_dists[idx]
        ld = ld_dists[idx]
        ratios = ld / (hd + 1e-12)
        return float(np.mean(ratios))

    # ------------------------------------------------------------------
    # Held-out Metrics
    # ------------------------------------------------------------------

    def rnx_auc(self, hd_data, ld_data, *, max_pairs_seed=0, max_n=1500):
        """Co-ranking RNX(K) area under the curve over log K (Lee & Verleysen).

        Deterministically subsamples to ``max_n`` rows for tractable O(n^2 log n)
        cost on large datasets; the subsample seed is fixed for reproducibility.
        Returns a scalar in roughly [0, 1]; higher is better.
        """
        import numpy as np
        from scipy.spatial.distance import squareform, pdist

        n = hd_data.shape[0]
        if n > max_n:
            rng = np.random.RandomState(max_pairs_seed)
            idx = rng.choice(n, max_n, replace=False)
            hd_data, ld_data = hd_data[idx], ld_data[idx]
            n = max_n
        if n < 4:
            return float("nan")

        hd_ranks = np.argsort(np.argsort(squareform(pdist(hd_data)), axis=1), axis=1)
        ld_ranks = np.argsort(np.argsort(squareform(pdist(ld_data)), axis=1), axis=1)

        # Co-ranking matrix Q[i, j]: count of pairs with HD-rank i+1 and LD-rank j+1.
        Q = np.zeros((n - 1, n - 1), dtype=np.int64)
        for a in range(n):
            r_hd = hd_ranks[a]
            r_ld = ld_ranks[a]
            for b in range(n):
                if a == b:
                    continue
                i = r_hd[b] - 1
                j = r_ld[b] - 1
                if 0 <= i < n - 1 and 0 <= j < n - 1:
                    Q[i, j] += 1

        rnx = []
        for K in range(1, n - 1):
            block = Q[:K, :K].sum()
            qnx_k = block / (K * n)
            rnx_k = ((n - 1) * qnx_k - K) / (n - 1 - K)
            rnx.append(rnx_k)
        rnx = np.asarray(rnx, dtype=float)

        # AUC weighted by 1/K (emphasis on small neighborhoods), normalised.
        ks = np.arange(1, len(rnx) + 1)
        weights = 1.0 / ks
        return float(np.sum(rnx * weights) / np.sum(weights))

    def shepard_pearson(self, hd_dists, ld_dists):
        """Pearson correlation of HD vs LD pairwise distances (Shepard goodness)."""
        import numpy as np
        iu = np.triu_indices_from(hd_dists, k=1)
        a = np.asarray(hd_dists)[iu]
        b = np.asarray(ld_dists)[iu]
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def knn_accuracy(self, ld_data, labels, k=10):
        """Leave-one-out kNN classifier accuracy on the 2-D embedding."""
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        if labels is None:
            return float("nan")
        labels = np.asarray(labels)
        n = ld_data.shape[0]
        kk = min(k, n - 1)
        if kk < 1:
            return float("nan")
        # Factorize labels to contiguous 0-based ints so bincount is well-defined
        # for arbitrary label encodings (strings, negatives, gaps).
        _, codes = np.unique(labels, return_inverse=True)
        # LOO via k+1 neighbours, dropping the self match (distance 0).
        clf = KNeighborsClassifier(n_neighbors=kk + 1)
        clf.fit(ld_data, codes)
        neigh = clf.kneighbors(ld_data, return_distance=False)[:, 1:]  # drop self
        preds = np.array([np.bincount(codes[row]).argmax() for row in neigh])
        return float((preds == codes).mean())

    def kmeans_nmi(self, ld_data, labels, n_clusters=None, seed=0):
        """NMI between KMeans clustering of the embedding and true labels."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
        if labels is None:
            return float("nan")
        labels = np.asarray(labels)
        k = n_clusters or len(np.unique(labels))
        if k < 2 or ld_data.shape[0] < k:
            return float("nan")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        pred = km.fit_predict(ld_data)
        return float(normalized_mutual_info_score(labels, pred))

    # ------------------------------------------------------------------
    # Local Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def trustworthiness(
        hd_dists: np.ndarray,
        ld_dists: np.ndarray,
        k: int,
        hd_ranks: Optional[np.ndarray] = None,
    ) -> float:
        """
        Trustworthiness T(k) — paper Eq. 4.
        Measures precision: how many LD k-NN are genuine HD neighbours.
        Higher (→ 1) = fewer false neighbours.
        """
        n = hd_dists.shape[0]
        if hd_ranks is None:
            hd_ranks = hd_dists.argsort(axis=1).argsort(axis=1)
        ld_nn = ld_dists.argsort(axis=1)[:, 1 : k + 1]

        penalty = 0.0
        for i in range(n):
            for j in ld_nn[i]:
                rank_in_hd = hd_ranks[i, j]
                if rank_in_hd > k:
                    penalty += rank_in_hd - k

        normalizer = n * k * (2 * n - 3 * k - 1)
        if normalizer == 0:
            return 1.0
        return float(1.0 - (2.0 / normalizer) * penalty)

    @staticmethod
    def continuity(
        hd_dists: np.ndarray,
        ld_dists: np.ndarray,
        k: int,
        hd_argsort: Optional[np.ndarray] = None,
    ) -> float:
        """
        Continuity C(k) — paper Eq. 5.
        Measures recall: how many HD k-NN remain neighbours in LD.
        Higher (→ 1) = original neighbours retained.
        """
        n = hd_dists.shape[0]
        ld_ranks = ld_dists.argsort(axis=1).argsort(axis=1)
        if hd_argsort is None:
            hd_argsort = hd_dists.argsort(axis=1)
        hd_nn = hd_argsort[:, 1 : k + 1]

        penalty = 0.0
        for i in range(n):
            for j in hd_nn[i]:
                rank_in_ld = ld_ranks[i, j]
                if rank_in_ld > k:
                    penalty += rank_in_ld - k

        normalizer = n * k * (2 * n - 3 * k - 1)
        if normalizer == 0:
            return 1.0
        return float(1.0 - (2.0 / normalizer) * penalty)

    # ------------------------------------------------------------------
    # Cluster Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def silhouette(
        ld_data: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 5000,
    ) -> Optional[float]:
        """Silhouette score for cluster separation quality."""
        if labels is None or len(np.unique(labels)) < 2:
            return None
        if len(ld_data) > sample_size:
            idx = np.random.choice(len(ld_data), sample_size, replace=False)
            ld_data = ld_data[idx]
            labels = labels[idx]
        return float(silhouette_score(ld_data, labels))

    # ------------------------------------------------------------------
    # LOF
    # ------------------------------------------------------------------

    @staticmethod
    def lof_statistics(ld_data: np.ndarray, n_neighbors: int = 20) -> dict:
        """
        LOF-based outlier detection (paper uses LOF in outlier forensics).
        Returns statistics suitable for the multi-modal input schema.
        """
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit(ld_data)
        scores = -lof.negative_outlier_factor_  # positive = more outlier-ish

        outlier_mask = scores > 1.5
        return {
            "LOF Median": round(float(np.median(scores)), 4),
            "LOF Std": round(float(np.std(scores)), 4),
            "LOF Outlier %": round(float(outlier_mask.sum() / len(scores) * 100), 2),
        }

    @staticmethod
    def compute_lof_scores(ld_data: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
        """Return per-point LOF scores (positive; >1.5 ≈ outlier)."""
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit(ld_data)
        return -lof.negative_outlier_factor_

    # ------------------------------------------------------------------
    # Pointwise distortion metrics (for Visual Forensics)
    # ------------------------------------------------------------------

    @staticmethod
    def pointwise_trustworthiness(
        hd_dists: np.ndarray, ld_dists: np.ndarray, k: int = 10
    ) -> np.ndarray:
        """
        Per-point trustworthiness: for each point, how many of its k LD
        neighbours are genuine HD neighbours.  Returns array of shape (n,)
        with values in [0, 1].
        """
        n = hd_dists.shape[0]
        hd_nn = hd_dists.argsort(axis=1)[:, 1:k + 1]  # true HD neighbours
        ld_nn = ld_dists.argsort(axis=1)[:, 1:k + 1]   # 2D neighbours

        scores = np.zeros(n)
        for i in range(n):
            hd_set = set(hd_nn[i])
            ld_set = set(ld_nn[i])
            scores[i] = len(hd_set & ld_set) / k
        return scores

    @staticmethod
    def pointwise_continuity(
        hd_dists: np.ndarray, ld_dists: np.ndarray, k: int = 10
    ) -> np.ndarray:
        """
        Per-point continuity: for each point, how many of its k HD
        neighbours remain among its k LD neighbours.  Returns (n,) in [0, 1].
        """
        n = hd_dists.shape[0]
        hd_nn = hd_dists.argsort(axis=1)[:, 1:k + 1]
        ld_nn = ld_dists.argsort(axis=1)[:, 1:k + 1]

        scores = np.zeros(n)
        for i in range(n):
            hd_set = set(hd_nn[i])
            ld_set = set(ld_nn[i])
            scores[i] = len(hd_set & ld_set) / k
        return scores

    @staticmethod
    def pointwise_stress(
        hd_dists: np.ndarray, ld_dists: np.ndarray
    ) -> np.ndarray:
        """
        Per-point local stress: mean squared distance distortion around
        each point.  Returns (n,) where higher = more distorted.
        """
        n = hd_dists.shape[0]
        # Normalise distances to [0, 1] for comparability
        hd_max = hd_dists.max() + 1e-12
        ld_max = ld_dists.max() + 1e-12
        hd_norm = hd_dists / hd_max
        ld_norm = ld_dists / ld_max

        diff_sq = (hd_norm - ld_norm) ** 2
        # Per-point: mean distortion to all other points
        scores = diff_sq.mean(axis=1)
        return scores

    def compute_pointwise_metrics(
        self,
        hd_data: np.ndarray,
        ld_data: np.ndarray,
        k: int = 10,
    ) -> dict:
        """
        Compute all per-point distortion metrics for the Visual Forensics phase.

        Returns dict with:
          - pointwise_trust: (n,) in [0, 1] — higher = better
          - pointwise_cont:  (n,) in [0, 1] — higher = better
          - pointwise_stress: (n,) — higher = more distorted
          - composite_distortion: (n,) in [0, 1] — 0 = perfect, 1 = worst
        """
        logger.info("Computing pairwise distances for pointwise metrics …")
        hd_dists = squareform(pdist(hd_data))
        ld_dists = squareform(pdist(ld_data))

        logger.info("Computing pointwise trustworthiness (k=%d) …", k)
        pw_trust = self.pointwise_trustworthiness(hd_dists, ld_dists, k)

        logger.info("Computing pointwise continuity (k=%d) …", k)
        pw_cont = self.pointwise_continuity(hd_dists, ld_dists, k)

        logger.info("Computing pointwise stress …")
        pw_stress = self.pointwise_stress(hd_dists, ld_dists)

        # Composite distortion: combine (1 - trust), (1 - cont), normalised stress
        stress_norm = pw_stress / (pw_stress.max() + 1e-12)
        composite = (
            0.35 * (1.0 - pw_trust)
            + 0.35 * (1.0 - pw_cont)
            + 0.30 * stress_norm
        )

        return {
            "pointwise_trust": pw_trust,
            "pointwise_cont": pw_cont,
            "pointwise_stress": pw_stress,
            "composite_distortion": composite,
        }
