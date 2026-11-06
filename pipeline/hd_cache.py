"""HD-side artifact cache.

In a sweep, the same dataset's HD-side metric inputs (pairwise distance matrix
and the argsort/ranks derived from it) are recomputed inside every
``metrics_engine.compute_all`` call — once per iteration × per goal — even
though they only depend on ``hd_data``, which never changes between calls.

This module persists those artifacts under
``outputs/_hd_cache/<dataset_key>/`` keyed by ``dataset_key`` and validated
against a sha256 of the HD data so a stale cache cannot leak across datasets
sharing a key by accident.

Files written per dataset:
    hd_dists.npy     — squareform(pdist(hd_data))   shape (n, n)
    hd_argsort.npy   — hd_dists.argsort(axis=1)     shape (n, n)
    hd_ranks.npy     — hd_argsort.argsort(axis=1)   shape (n, n)
    meta.json        — {sha256, n, d, dtype}

The matrices are O(n²) floats/ints. For the sweep's typical n (1k-10k) this
fits comfortably on disk and dwarfs the recompute cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np


class HDArtifactCache:
    """Load-or-compute HD pairwise distance + argsort/rank matrices."""

    def __init__(self, cache_root: str, dataset_key: Optional[str]):
        self.cache_root = cache_root
        self.dataset_key = dataset_key
        self.cache_dir: Optional[str] = (
            os.path.join(cache_root, dataset_key) if dataset_key else None
        )
        # In-process memo so a single planner run only touches disk once even
        # if compute_all is called many times.
        self._memo: Optional[dict] = None

    @staticmethod
    def _hash(hd_data: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(hd_data.shape.__repr__().encode("utf-8"))
        h.update(str(hd_data.dtype).encode("utf-8"))
        # Use the underlying buffer; np.ascontiguousarray ensures a stable
        # byte layout regardless of how ``hd_data`` was sliced upstream.
        h.update(np.ascontiguousarray(hd_data).tobytes())
        return h.hexdigest()

    def get_or_compute(self, hd_data: np.ndarray) -> dict:
        """Return ``{"hd_dists", "hd_argsort", "hd_ranks"}`` for ``hd_data``.

        The memo / disk cache is only returned when the sha256 of ``hd_data``
        matches what was cached. A non-matching call (e.g. a subsampled patch
        in ``_compute_patch_reliability``) recomputes fresh artifacts and is
        NEVER allowed to overwrite the canonical full-dataset cache — that
        used to corrupt the memo and produce
        ``IndexError: index … is out of bounds for axis 1 with size …`` in
        ``trustworthiness`` on the next full-data call.
        """
        expected_hash = self._hash(hd_data)

        if self._memo is not None and self._memo.get("_sha256") == expected_hash:
            return self._memo

        cache_dir = self.cache_dir
        meta_path = (
            os.path.join(cache_dir, "meta.json") if cache_dir else None
        )

        if cache_dir and meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if (
                    meta.get("sha256") == expected_hash
                    and meta.get("n") == int(hd_data.shape[0])
                    and meta.get("d") == int(hd_data.shape[1])
                ):
                    hd_dists = np.load(os.path.join(cache_dir, "hd_dists.npy"))
                    hd_argsort = np.load(os.path.join(cache_dir, "hd_argsort.npy"))
                    hd_ranks = np.load(os.path.join(cache_dir, "hd_ranks.npy"))
                    artifacts = {
                        "hd_dists": hd_dists,
                        "hd_argsort": hd_argsort,
                        "hd_ranks": hd_ranks,
                        "_sha256": expected_hash,
                    }
                    if self._memo is None:
                        self._memo = artifacts
                    print(
                        f"  ⚡ Loaded HD artifacts from cache ({cache_dir}); "
                        "skipping pdist + argsort.",
                        flush=True,
                    )
                    return artifacts
            except Exception as exc:
                print(
                    f"  ⚠️  HD cache load failed ({exc}) — recomputing.",
                    flush=True,
                )

        # Compute (deferred import keeps this module cheap to import).
        from scipy.spatial.distance import pdist, squareform
        hd_dists = squareform(pdist(hd_data))
        hd_argsort = hd_dists.argsort(axis=1)
        hd_ranks = hd_argsort.argsort(axis=1)
        artifacts = {
            "hd_dists": hd_dists,
            "hd_argsort": hd_argsort,
            "hd_ranks": hd_ranks,
            "_sha256": expected_hash,
        }

        # Only treat the FIRST computed artifact as canonical: memo it and
        # persist it. Later calls with different ``hd_data`` (subsampled
        # patches, neighborhood grids, etc.) get a fresh result without
        # poisoning the memo or the on-disk cache for the next sweep cell.
        if self._memo is None:
            self._memo = artifacts
            if cache_dir is not None:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(os.path.join(cache_dir, "hd_dists.npy"), hd_dists)
                    np.save(os.path.join(cache_dir, "hd_argsort.npy"), hd_argsort)
                    np.save(os.path.join(cache_dir, "hd_ranks.npy"), hd_ranks)
                    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
                        json.dump({
                            "sha256": expected_hash,
                            "n": int(hd_data.shape[0]),
                            "d": int(hd_data.shape[1]),
                            "dtype": str(hd_data.dtype),
                        }, f, indent=2)
                except Exception as exc:
                    print(
                        f"  ⚠️  HD cache write failed ({exc}) — continuing without persistence.",
                        flush=True,
                    )

        return artifacts
