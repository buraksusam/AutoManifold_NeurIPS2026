"""NovelV2SweepPlanner — lightweight sweep variant of NovelPlannerV2.

Differences vs NovelPlannerV2:
  - Phase 0 caches baseline metrics + embeddings per dataset under
    ``outputs/_baseline_cache/<dataset_key>/``. On a cache hit the heavy
    t-SNE/UMAP/PaCMAP + metric computations are skipped; only the
    intent-dependent scoring + winner selection + profile assembly run.
  - Phase 6b (Mathematical Representation & Pseudocode) is skipped.
  - Phase 7 (Visual Forensics) is skipped.

The pipeline still runs Phases 0-5 (baseline → characterize → difficulty →
compose → novelty → synthesize) and Phase 6 (implement & evaluate, with
iterative optimization). That's the workflow the sweep needs: generate the
novel algorithm, optimize, and evaluate — nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from .hd_cache import HDArtifactCache
from .config import OUTPUT_ROOT
from .planner_novel_v2 import NovelPlannerV2


class _HDCachedMetricsEngine:
    """Proxy around ``QuantitativeAnalystEngine`` that injects cached HD-side
    artifacts (pdist matrix, argsort, ranks) into every ``compute_all`` call.

    The cache is keyed by ``dataset_key`` and validated by the sha256 of
    ``hd_data`` on first use; subsequent calls hit the in-process memo.
    All other engine methods are forwarded unchanged via ``__getattr__``.
    """

    def __init__(self, inner, hd_cache: HDArtifactCache):
        self._inner = inner
        self._hd_cache = hd_cache

    def compute_all(self, hd_data, ld_data, **kwargs):
        # Caller may already pass artifacts in (e.g. base class explicit pass);
        # if so, respect them. Otherwise inject from cache.
        if (
            kwargs.get("hd_dists") is None
            and kwargs.get("hd_argsort") is None
            and kwargs.get("hd_ranks") is None
        ):
            artifacts = self._hd_cache.get_or_compute(hd_data)
            kwargs["hd_dists"] = artifacts["hd_dists"]
            kwargs["hd_argsort"] = artifacts["hd_argsort"]
            kwargs["hd_ranks"] = artifacts["hd_ranks"]
        return self._inner.compute_all(hd_data, ld_data, **kwargs)

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails on self; forward
        # everything else to the wrapped engine.
        return getattr(self._inner, name)


class NovelV2SweepPlanner(NovelPlannerV2):
    """novelv2 variant tuned for sweep experiments."""

    BASELINE_CACHE_ROOT = os.path.join(OUTPUT_ROOT, "_baseline_cache")
    HD_CACHE_ROOT = os.path.join(OUTPUT_ROOT, "_hd_cache")

    def __init__(self, config, dataset_key: Optional[str] = None):
        super().__init__(config)
        from .agents.agent_8_synthesizer import AlgorithmSynthesizerAgent

        self.mode_name = "novelv2_sweep"
        self.template_name = "DR_Template_v2"
        self.agent8 = AlgorithmSynthesizerAgent(self.llm, mode="novel_v2")
        self.dataset_key = dataset_key

        # Wrap the metrics engine so every compute_all call reuses the cached
        # HD pdist/argsort/ranks for this dataset. First call populates both
        # the in-process memo and the on-disk cache; later calls (same goal,
        # next iteration; or next goal in this process) skip pdist entirely.
        hd_cache = HDArtifactCache(self.HD_CACHE_ROOT, dataset_key=dataset_key)
        self.metrics_engine = _HDCachedMetricsEngine(self.metrics_engine, hd_cache)

    # --------------------------------------------------------------
    # Phase 0 — cached baselines
    # --------------------------------------------------------------

    def _phase0_baseline_calibration(
        self, data: np.ndarray, labels: Optional[np.ndarray],
    ) -> None:
        intent = self._infer_intent(self.config.primary_goal)
        methods = ["tsne", "umap", "pacmap"]

        cache_key = self.dataset_key
        cache_dir = (
            os.path.join(self.BASELINE_CACHE_ROOT, cache_key)
            if cache_key else None
        )
        cache_metrics_path = (
            os.path.join(cache_dir, "metrics_per_method.json")
            if cache_dir else None
        )

        # Try cache first
        if cache_dir and cache_metrics_path and os.path.exists(cache_metrics_path):
            try:
                with open(cache_metrics_path) as f:
                    metrics_per_method = json.load(f)
                embeddings_path: Dict[str, str] = {}
                ok = True
                for m in methods:
                    src = os.path.join(cache_dir, f"baseline_{m}.npy")
                    if os.path.exists(src):
                        dst = os.path.join(self.output_dir, f"baseline_{m}.npy")
                        if os.path.abspath(src) != os.path.abspath(dst):
                            try:
                                shutil.copy2(src, dst)
                            except Exception:
                                pass
                        embeddings_path[m] = dst
                    else:
                        ok = False
                        break
                # Verify each method's metrics dict carries the keys we need
                for m in methods:
                    mm = metrics_per_method.get(m, {})
                    if "error" in mm:
                        continue
                    for required in ("Trustworthiness", "Continuity", "Stress",
                                     "Spearman", "Silhouette"):
                        if required not in mm:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    print(
                        f"  ⚡ Loaded baseline metrics from cache "
                        f"({cache_dir}); skipping DR + metrics computation.",
                        flush=True,
                    )
                    self._build_and_save_baseline_profile(
                        metrics_per_method, embeddings_path, methods, intent,
                    )
                    return
                print(
                    "  ⚠️  Baseline cache incomplete — recomputing.",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  ⚠️  Baseline cache load failed ({exc}) — recomputing.",
                    flush=True,
                )

        # Cache miss — run the full Phase 0
        super()._phase0_baseline_calibration(data, labels)

        # Persist cache for the next sweep cell on the same dataset
        if cache_dir is None:
            return
        profile = self.context.get("baseline_profile") or {}
        if not profile:
            return
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "metrics_per_method.json"), "w") as f:
                json.dump(profile.get("metrics_per_method", {}), f,
                          indent=2, default=str)
            for m, src in (profile.get("embeddings_path") or {}).items():
                dst = os.path.join(cache_dir, f"baseline_{m}.npy")
                try:
                    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
                except Exception as exc:
                    print(
                        f"  ⚠️  Could not cache baseline_{m}.npy: {exc}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"  ⚠️  Baseline cache write failed: {exc}", flush=True)

    # --------------------------------------------------------------
    # Phases 6b + 7 — disabled
    # --------------------------------------------------------------

    def _phase_math_pseudocode(
        self,
        algorithm_name: str,
        code_str: str,
        default_params: Dict[str, Any],
        description: str = "",
        output_prefix: str = "novel",
    ) -> Dict[str, Any]:
        print(
            "  ⏭  Math pseudocode skipped (novelv2_sweep mode).",
            flush=True,
        )
        return {"skipped": "novelv2_sweep_mode"}

    def _phase7_visual_forensics(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        print(
            "  ⏭  Visual forensics skipped (novelv2_sweep mode).",
            flush=True,
        )
        return {"skipped": "novelv2_sweep_mode"}
