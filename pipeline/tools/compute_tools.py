"""Compute tools — wrap deterministic engines + helpers as agent-loop tools.

Each handler signature is ``handler(args: dict, session_state) -> dict``.
The session_state instance is defined in :mod:`pipeline.agent_loop` and holds
HD data, labels, embeddings (under string handles), engine instances, and a
running log of code-validation attempts.

All handler returns are JSON-serializable: numpy arrays are stored on the
session_state under handles and only summary statistics are returned.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — baseline scoring (mirrors AutoManifoldPlanner._score_baseline / _infer_intent)
# ---------------------------------------------------------------------------

def _infer_intent(primary_goal: str) -> str:
    g = (primary_goal or "").lower()
    if "trustworthiness" in g:
        return "trustworthiness"
    if "continuity" in g:
        return "continuity"
    if "spearman" in g:
        return "spearman"
    if "silhouette" in g:
        return "silhouette"
    if "lof" in g:
        return "lof"
    if "stress" in g:
        return "stress"
    if "local" in g:
        return "local"
    if "global" in g or "topology" in g or "skeleton" in g:
        return "global"
    return "balanced"


def _score_baseline(metrics: Dict[str, Any], intent: str) -> float:
    T = float(metrics.get("Trustworthiness", 0.0) or 0.0)
    C = float(metrics.get("Continuity", 0.0) or 0.0)
    S = float(metrics.get("Stress", 1.0) or 1.0)
    Sp = float(metrics.get("Spearman", 0.0) or 0.0)
    Sil = float(metrics.get("Silhouette", 0.0) or 0.0)
    LOFpct = float(metrics.get("LOF_Outlier_pct", 100.0) or 100.0)
    s_norm = 1.0 - min(max(S, 0.0), 1.0)
    sil_norm = 0.5 * (max(-1.0, min(1.0, Sil)) + 1.0)
    lof_norm = 1.0 - min(max(LOFpct, 0.0), 100.0) / 100.0
    if intent == "local":
        return 0.7 * T + 0.3 * C
    if intent == "global":
        return 0.5 * Sp + 0.5 * s_norm
    if intent == "trustworthiness":
        return T
    if intent == "continuity":
        return C
    if intent == "spearman":
        return max(0.0, min(1.0, Sp))
    if intent == "silhouette":
        return sil_norm
    if intent == "stress":
        return s_norm
    if intent == "lof":
        return lof_norm
    return 0.25 * (T + C + Sp + s_norm)


def _flatten_metrics(nested: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested compute_all() result into the snapshot dict
    used by _score_baseline."""
    flat: Dict[str, Any] = {}
    local = nested.get("Neighborhood Preservation", {}) or {}
    glob = nested.get("Global Structure & HD<->2D Correlation", {}) or {}
    cluster = nested.get("Clustering / Class Separation", {}) or {}
    outlier = nested.get("Outlier Behavior", {}) or {}
    for k, v in local.items():
        if k.startswith("Trustworthiness") and "AUC" not in k and "curve" not in k:
            flat.setdefault("Trustworthiness", v)
        if k.startswith("Continuity") and "AUC" not in k and "curve" not in k:
            flat.setdefault("Continuity", v)
    flat["Trustworthiness_AUC"] = local.get("Trustworthiness AUC")
    flat["Continuity_AUC"] = local.get("Continuity AUC")
    flat["k_sweep"] = local.get("k_sweep")
    flat["Trustworthiness_curve"] = local.get("Trustworthiness curve")
    flat["Continuity_curve"] = local.get("Continuity curve")
    flat["Stress"] = glob.get("Stress")
    flat["Spearman"] = glob.get("Spearman")
    flat["Silhouette"] = cluster.get("Silhouette") or cluster.get("Silhouette Score")
    flat["LOF_Outlier_pct"] = outlier.get("LOF_Outlier_pct") or outlier.get("LOF Outlier %")
    return flat


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _tool_compute_baseline_calibration(args: Dict[str, Any], state) -> Dict[str, Any]:
    methods = args.get("methods") or ["tsne", "umap", "pacmap"]
    intent = _infer_intent(state.primary_goal)
    metrics_per_method: Dict[str, Dict[str, float]] = {}
    handles: Dict[str, str] = {}
    for method in methods:
        try:
            emb = state.dr_engine.compute_embedding(state.data, method=method)
            nested = state.metrics_engine.compute_all(
                hd_data=state.data, ld_data=emb, labels=state.labels,
                k=state.metrics_k, lof_neighbors=state.lof_neighbors,
            )
            snap = _flatten_metrics(nested)
            metrics_per_method[method] = {
                "Trustworthiness": float(snap.get("Trustworthiness") or 0.0),
                "Continuity": float(snap.get("Continuity") or 0.0),
                "Trustworthiness_AUC": float(snap.get("Trustworthiness_AUC") or 0.0),
                "Continuity_AUC": float(snap.get("Continuity_AUC") or 0.0),
                "k_sweep": snap.get("k_sweep"),
                "Trustworthiness_curve": snap.get("Trustworthiness_curve"),
                "Continuity_curve": snap.get("Continuity_curve"),
                "Stress": float(snap.get("Stress") or 1.0),
                "Spearman": float(snap.get("Spearman") or 0.0),
                "Silhouette": float(snap.get("Silhouette") or 0.0),
            }
            handle = f"baseline_{method}"
            state.embeddings[handle] = emb
            state.metrics_cache[handle] = nested
            handles[method] = handle
        except Exception as exc:
            metrics_per_method[method] = {"error": str(exc)}

    scored = {
        m: _score_baseline(v, intent)
        for m, v in metrics_per_method.items()
        if "error" not in v
    }
    if not scored:
        return {
            "error": "All baselines failed",
            "metrics_per_method": metrics_per_method,
        }
    winner = max(scored, key=scored.get)
    winner_metrics = metrics_per_method[winner]
    beat_targets = {
        "Trustworthiness": winner_metrics["Trustworthiness"],
        "Continuity": winner_metrics["Continuity"],
        "Trustworthiness_AUC": winner_metrics.get("Trustworthiness_AUC", 0.0),
        "Continuity_AUC": winner_metrics.get("Continuity_AUC", 0.0),
        "Stress": winner_metrics["Stress"],
        "Spearman": winner_metrics["Spearman"],
        "tolerance": 0.005,
    }
    profile = {
        "primary_goal": state.primary_goal,
        "intent": intent,
        "methods_evaluated": list(methods),
        "metrics_per_method": metrics_per_method,
        "embedding_handles": handles,
        "winner": {
            "name": winner,
            "primary_goal_score": round(scored[winner], 4),
            "all_scores": {m: round(s, 4) for m, s in scored.items()},
        },
        "beat_targets": beat_targets,
    }
    state.baseline_profile = profile
    return profile


def _tool_compute_pca_summary(args: Dict[str, Any], state) -> Dict[str, Any]:
    from sklearn.decomposition import PCA
    pca = PCA().fit(state.data)
    evr = pca.explained_variance_ratio_
    n_dim = state.data.shape[1]
    explained = {}
    for k in (10, 20, 50):
        if k <= n_dim:
            explained[str(k)] = round(float(np.sum(evr[:k])), 4)
        else:
            explained[str(k)] = round(float(np.sum(evr)), 4)
    # Pseudo-MLE: smallest k where cumulative > 0.9 (truncated dim estimate)
    cumsum = np.cumsum(evr)
    intrinsic = int(np.searchsorted(cumsum, 0.9) + 1) if cumsum[-1] >= 0.9 else int(n_dim)
    state.metrics_cache["pca_summary"] = {
        "explained_variance_ratio": [float(x) for x in evr.tolist()],
    }
    return {
        "pca_explained_variance": explained,
        "intrinsic_dimensionality_estimate": intrinsic,
        "n_features": int(n_dim),
    }


def _tool_compute_patch_scores(args: Dict[str, Any], state) -> Dict[str, Any]:
    from sklearn.cluster import MiniBatchKMeans

    n = state.data.shape[0]
    target_size = getattr(state, "target_patch_size", None) or 200
    n_patches = max(1, n // target_size)
    km = MiniBatchKMeans(
        n_clusters=n_patches, random_state=42, batch_size=min(1024, n), n_init=3,
    )
    patch_labels = km.fit_predict(state.data)

    results: List[Dict[str, Any]] = []
    patch_reliability: Dict[int, float] = {}

    for pid in range(n_patches):
        mask = patch_labels == pid
        patch_data = state.data[mask]
        if patch_data.shape[0] < 5:
            continue
        patch_lbl = state.labels[mask] if state.labels is not None else None
        scores: Dict[str, Any] = {"patch_id": int(pid), "size": int(patch_data.shape[0])}
        rel_vals = []
        for method in ["tsne", "umap"]:
            try:
                emb = state.dr_engine.compute_embedding(
                    patch_data, method=method,
                    n_pcs=min(20, patch_data.shape[1]),
                )
                m = state.metrics_engine.compute_all(
                    hd_data=patch_data, ld_data=emb, labels=patch_lbl,
                    k=state.metrics_k, lof_neighbors=state.lof_neighbors,
                )
                local_m = m.get("Neighborhood Preservation", {}) or {}
                global_m = m.get("Global Structure & HD<->2D Correlation", {}) or {}
                t = next((v for k2, v in local_m.items() if k2.startswith("Trustworthiness")), None)
                c = next((v for k2, v in local_m.items() if k2.startswith("Continuity")), None)
                s = global_m.get("Stress")
                if t is not None and c is not None and s is not None:
                    rel_vals.append((float(t) + float(c) + (1.0 - float(s))) / 3.0)
                    scores[method] = {
                        "Trustworthiness": float(t),
                        "Continuity": float(c),
                        "Stress": float(s),
                    }
            except Exception as exc:
                scores[method] = {"error": str(exc)}
        if rel_vals:
            patch_reliability[int(pid)] = float(np.mean(rel_vals))
        results.append(scores)

    observed = list(patch_reliability.values())
    fallback = float(np.median(observed)) if observed else 0.5
    point_reliability = np.array(
        [patch_reliability.get(int(pid), fallback) for pid in patch_labels],
        dtype=float,
    )
    q_low = float(np.percentile(point_reliability, 33))
    q_high = float(np.percentile(point_reliability, 67))
    if q_high - q_low < 0.02:
        med = float(np.median(point_reliability))
        q_low = med - 0.01
        q_high = med + 0.01
    point_difficulty = np.where(
        point_reliability >= q_high, "E",
        np.where(point_reliability <= q_low, "H", "M"),
    ).tolist()

    state.point_difficulty = point_difficulty
    state.point_reliability = point_reliability.tolist()
    return {
        "patch_scores": results,
        "point_difficulty_summary": {
            "H_count": int(point_difficulty.count("H")),
            "M_count": int(point_difficulty.count("M")),
            "E_count": int(point_difficulty.count("E")),
        },
        "reliability_quantiles": {
            "q_low": q_low,
            "q_high": q_high,
            "median": float(np.median(point_reliability)),
        },
    }


def _tool_compute_neighborhood_grid(args: Dict[str, Any], state) -> Dict[str, Any]:
    n = state.data.shape[0]
    sample_size = min(10, max(1, n // 50))
    rng = np.random.default_rng(42)
    target_indices = rng.choice(n, size=sample_size, replace=False)
    k_values = args.get("k_grid") or [20, 50, 100, 200, 500]
    results: List[Dict[str, Any]] = []
    for idx in target_indices:
        dists = np.linalg.norm(state.data - state.data[idx], axis=1)
        for k in k_values:
            if k > n:
                continue
            nn_idx = np.argsort(dists)[:k]
            patch = state.data[nn_idx]
            patch_lbl = state.labels[nn_idx] if state.labels is not None else None
            try:
                emb = state.dr_engine.compute_embedding(
                    patch, method="tsne",
                    n_pcs=min(20, patch.shape[1]),
                )
                m = state.metrics_engine.compute_all(
                    hd_data=patch, ld_data=emb, labels=patch_lbl,
                )
                results.append({
                    "target_point": int(idx),
                    "k": int(k),
                    "T": m.get("Trustworthiness"),
                    "C": m.get("Continuity"),
                    "S": m.get("Stress-1"),
                })
            except Exception as exc:
                results.append({"target_point": int(idx), "k": int(k), "error": str(exc)})
    return {"grid_results": results, "sampled_targets": target_indices.tolist(),
            "k_values": list(k_values)}


def _tool_validate_custom_code(args: Dict[str, Any], state) -> Dict[str, Any]:
    code = args.get("code") or ""
    default_params = args.get("default_params") or {}
    if not code.strip():
        return {"ok": False, "error_message": "empty code"}

    # Subsample to ≤500 rows for fast validation
    data = state.data
    labels = state.labels
    n = data.shape[0]
    if n > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 500, replace=False)
        test_data = data[idx]
        test_labels = labels[idx] if labels is not None else None
    else:
        test_data = data
        test_labels = labels

    namespace: Dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    try:
        exec(compile(code, "<novel_algorithm>", "exec"), namespace)
    except Exception as exc:
        attempt = {"ok": False, "error_message": f"Compilation error: {exc}", "code": code}
        state.code_attempts.append(attempt)
        return {"ok": False, "error_message": attempt["error_message"]}

    fn = namespace.get("custom_embedding")
    if fn is None or not callable(fn):
        attempt = {"ok": False, "error_message": "no callable 'custom_embedding'", "code": code}
        state.code_attempts.append(attempt)
        return attempt

    try:
        result = fn(test_data, test_labels, default_params)
    except Exception as exc:
        attempt = {"ok": False, "error_message": f"Runtime error: {exc}", "code": code}
        state.code_attempts.append(attempt)
        return {"ok": False, "error_message": attempt["error_message"]}

    if not isinstance(result, np.ndarray) or result.ndim != 2 or result.shape[1] != 2:
        attempt = {
            "ok": False,
            "error_message": f"unexpected output shape: {getattr(result, 'shape', type(result))}",
            "code": code,
        }
        state.code_attempts.append(attempt)
        return {"ok": False, "error_message": attempt["error_message"]}
    if np.isnan(result).any() or np.isinf(result).any():
        attempt = {"ok": False, "error_message": "output contains NaN or Inf", "code": code}
        state.code_attempts.append(attempt)
        return {"ok": False, "error_message": attempt["error_message"]}

    state.code_attempt_count = getattr(state, "code_attempt_count", 0) + 1
    handle = f"novel_attempt_{state.code_attempt_count}"
    state.embeddings[handle] = result
    sample_metrics: Dict[str, Any] = {}
    try:
        nested = state.metrics_engine.compute_all(
            hd_data=test_data, ld_data=result, labels=test_labels,
            k=min(state.metrics_k, max(2, test_data.shape[0] // 4)),
            lof_neighbors=min(state.lof_neighbors, max(2, test_data.shape[0] // 4)),
        )
        sample_metrics = _flatten_metrics(nested)
    except Exception as exc:
        sample_metrics = {"error": str(exc)}

    state.code_attempts.append({"ok": True, "code": code, "handle": handle})
    return {
        "ok": True,
        "embedding_handle": handle,
        "embedding_shape": list(result.shape),
        "sample_metrics": sample_metrics,
    }


def _tool_compute_pointwise_metrics(args: Dict[str, Any], state) -> Dict[str, Any]:
    handle = args.get("embedding_handle") or ""
    if handle not in state.embeddings:
        return {"error": f"unknown embedding handle '{handle}'"}
    emb = state.embeddings[handle]
    pw = state.metrics_engine.compute_pointwise_metrics(
        hd_data=state.data, ld_data=emb, k=state.metrics_k,
    )
    composite = pw["composite_distortion"]
    pw_trust = pw["pointwise_trust"]
    pw_cont = pw["pointwise_cont"]
    summary = {
        "mean_distortion": float(composite.mean()),
        "median_distortion": float(np.median(composite)),
        "p90_distortion": float(np.percentile(composite, 90)),
        "p95_distortion": float(np.percentile(composite, 95)),
        "mean_trust": float(pw_trust.mean()),
        "mean_cont": float(pw_cont.mean()),
    }
    state.metrics_cache[f"pointwise_{handle}"] = summary
    return {"embedding_handle": handle, "summary": summary}


def _tool_compute_hierarchy_diff(args: Dict[str, Any], state) -> Dict[str, Any]:
    handle = args.get("embedding_handle") or ""
    if handle not in state.embeddings:
        return {"error": f"unknown embedding handle '{handle}'"}
    emb = state.embeddings[handle]
    try:
        dendrograms = state.clustering_engine.compute_dendrograms(
            hd_data=state.data, ld_data=emb, labels=state.labels,
        )
        diff = state.clustering_engine.compute_hierarchy_diff(
            dendrograms, label_names=state.label_names, dr_method="custom",
        )
    except Exception as exc:
        return {"error": f"hierarchy diff failed: {exc}"}
    return {
        "embedding_handle": handle,
        "robinson_foulds_distance": diff.get("robinson_foulds_distance"),
        "cophenetic_correlation": diff.get("cophenetic_correlation"),
        "n_topology_changes": len(diff.get("topology_changes", []) or []),
        "summary": diff.get("summary"),
    }


def _tool_compute_class_distortion_stats(args: Dict[str, Any], state) -> Dict[str, Any]:
    handle = args.get("embedding_handle") or ""
    if handle not in state.embeddings:
        return {"error": f"unknown embedding handle '{handle}'"}
    if state.labels is None:
        return {"error": "no labels available"}
    emb = state.embeddings[handle]
    pw = state.metrics_engine.compute_pointwise_metrics(
        hd_data=state.data, ld_data=emb, k=state.metrics_k,
    )
    composite = pw["composite_distortion"]
    pw_trust = pw["pointwise_trust"]
    pw_cont = pw["pointwise_cont"]
    p90 = np.percentile(composite, 90)
    out: Dict[str, Any] = {}
    label_names = state.label_names or {}
    for lab in np.unique(state.labels):
        mask = state.labels == lab
        name = label_names.get(int(lab), str(int(lab)))
        out[name] = {
            "n_points": int(mask.sum()),
            "mean_distortion": round(float(composite[mask].mean()), 4),
            "mean_trust": round(float(pw_trust[mask].mean()), 4),
            "mean_cont": round(float(pw_cont[mask].mean()), 4),
            "p90_distortion": round(float(np.percentile(composite[mask], 90)), 4),
            "pct_high_distortion": round(
                float((composite[mask] > p90).sum() / max(1, mask.sum()) * 100), 1,
            ),
        }
    return {"embedding_handle": handle, "per_class": out}


def _tool_finalize_design(args: Dict[str, Any], state) -> Dict[str, Any]:
    state.final_report = args.get("final_report", {}) or args
    state.done = True
    return {"acknowledged": True}


# ---------------------------------------------------------------------------
# Schemas + registration
# ---------------------------------------------------------------------------

_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "compute_baseline_calibration": {
        "description": (
            "Run t-SNE / UMAP / PaCMAP baseline DR methods on the dataset, "
            "score them against the user's primary_goal, and return the winner "
            "plus all per-method metrics. Stores embeddings under handles "
            "'baseline_<method>'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Methods to run (default: tsne, umap, pacmap).",
                },
            },
            "required": [],
        },
    },
    "compute_pca_summary": {
        "description": (
            "Compute PCA explained-variance summary at n_components ∈ {10, 20, 50} "
            "and return a pseudo-MLE intrinsic-dimensionality estimate."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "compute_patch_scores": {
        "description": (
            "Cluster the data into patches and compute patch-level reliability "
            "(local DR scores) plus per-point H/M/E difficulty assignments."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "compute_neighborhood_grid": {
        "description": (
            "Grid search over neighborhood radii k ∈ k_grid: for each, embed a "
            "k-NN ball around sample points and compute T/C/S to estimate the "
            "maximal embeddable neighborhood."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "k_grid": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of k values (default: 20, 50, 100, 200, 500).",
                },
            },
            "required": [],
        },
    },
    "validate_custom_code": {
        "description": (
            "Compile and execute a synthesized `custom_embedding(data, labels, params)` "
            "function on a 500-row subsample. Returns ok=true with embedding handle on "
            "success, ok=false with error_message on failure (compile/runtime/shape/NaN)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python source defining custom_embedding."},
                "default_params": {"type": "object", "description": "Hyperparameters dict."},
            },
            "required": ["code"],
        },
    },
    "compute_pointwise_metrics": {
        "description": (
            "Compute per-point trustworthiness, continuity, stress, and composite "
            "distortion for a previously-stored embedding handle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "embedding_handle": {"type": "string", "description": "e.g. 'baseline_tsne' or 'novel_attempt_1'."},
            },
            "required": ["embedding_handle"],
        },
    },
    "compute_hierarchy_diff": {
        "description": (
            "Compute HD vs 2D dendrograms for the embedding handle and return Robinson-Foulds "
            "distance, cophenetic correlation, and topology-change summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"embedding_handle": {"type": "string"}},
            "required": ["embedding_handle"],
        },
    },
    "compute_class_distortion_stats": {
        "description": (
            "Compute per-class distortion summary (n_points, mean/p90 distortion, "
            "pct_high_distortion) for a stored embedding handle. Requires labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"embedding_handle": {"type": "string"}},
            "required": ["embedding_handle"],
        },
    },
    "finalize_design": {
        "description": (
            "Call this when the design is complete to stop the agent loop and "
            "persist the final report. Pass a `final_report` object summarising "
            "the chosen algorithm name, code (or handle), key metrics, and "
            "novelty assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "final_report": {
                    "type": "object",
                    "description": "Summary report object (algorithm name, code, metrics, novelty, etc.).",
                },
            },
            "required": ["final_report"],
        },
    },
}


_HANDLERS = {
    "compute_baseline_calibration": _tool_compute_baseline_calibration,
    "compute_pca_summary": _tool_compute_pca_summary,
    "compute_patch_scores": _tool_compute_patch_scores,
    "compute_neighborhood_grid": _tool_compute_neighborhood_grid,
    "validate_custom_code": _tool_validate_custom_code,
    "compute_pointwise_metrics": _tool_compute_pointwise_metrics,
    "compute_hierarchy_diff": _tool_compute_hierarchy_diff,
    "compute_class_distortion_stats": _tool_compute_class_distortion_stats,
}


def register_compute_tools(registry: ToolRegistry) -> None:
    """Register every compute tool (excluding finalize_design)."""
    for name, handler in _HANDLERS.items():
        registry.register(name, _SCHEMAS[name], handler)


def register_finalize_tool(registry: ToolRegistry) -> None:
    """Register the control tool ``finalize_design``."""
    registry.register(
        "finalize_design",
        _SCHEMAS["finalize_design"],
        _tool_finalize_design,
    )
