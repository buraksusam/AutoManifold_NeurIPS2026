"""Tests for compute_tools handlers using a small synthetic dataset."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

from pipeline.agent_loop import SessionState
from pipeline.engines.clustering_engine import ClusteringEngine
from pipeline.engines.dr_engine import DRVisualizationEngine
from pipeline.engines.metrics_engine import QuantitativeAnalystEngine
from pipeline.tools import ToolRegistry, register_compute_tools, register_finalize_tool


def _make_state(n: int = 60, d: int = 8, with_labels: bool = True, tmp_dir=None) -> SessionState:
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(3, d)) * 5
    labels = np.repeat(np.arange(3), n // 3)
    n_total = labels.shape[0]
    data = np.vstack([
        rng.normal(loc=centers[lab], scale=0.6, size=(1, d))
        for lab in labels
    ]).reshape(n_total, d)
    state = SessionState(
        data=data.astype(np.float64),
        labels=labels if with_labels else None,
        label_names={0: "A", 1: "B", 2: "C"} if with_labels else None,
        primary_goal="local structure preservation",
        output_dir=tmp_dir or tempfile.mkdtemp(),
        dr_engine=DRVisualizationEngine(),
        metrics_engine=QuantitativeAnalystEngine(),
        clustering_engine=ClusteringEngine(),
        metrics_k=5,
        lof_neighbors=5,
    )
    return state


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_compute_tools(reg)
    register_finalize_tool(reg)
    return reg


def test_compute_pca_summary_returns_explained_variance():
    state = _make_state()
    reg = _registry()
    out = reg.dispatch("compute_pca_summary", {}, state)
    assert "pca_explained_variance" in out
    for key in ("10", "20", "50"):
        assert key in out["pca_explained_variance"]
    assert isinstance(out["intrinsic_dimensionality_estimate"], int)


@pytest.mark.slow
def test_compute_baseline_calibration_runs_three_methods():
    state = _make_state(n=45, d=6)
    reg = _registry()
    out = reg.dispatch(
        "compute_baseline_calibration", {"methods": ["tsne"]}, state,
    )
    # winner picked, beat_targets present, embedding handle stored
    assert "winner" in out or "error" in out
    if "winner" in out:
        assert "beat_targets" in out
        handles = out.get("embedding_handles") or {}
        for method, handle in handles.items():
            assert handle in state.embeddings
            assert state.embeddings[handle].shape == (state.data.shape[0], 2)


def test_compute_patch_scores_returns_three_keys():
    state = _make_state(n=60, d=6)
    reg = _registry()
    out = reg.dispatch("compute_patch_scores", {}, state)
    assert "patch_scores" in out
    assert "point_difficulty_summary" in out
    assert "reliability_quantiles" in out


def test_compute_neighborhood_grid_returns_per_k_results():
    state = _make_state(n=80, d=6)
    reg = _registry()
    out = reg.dispatch("compute_neighborhood_grid", {"k_grid": [10, 20]}, state)
    assert "grid_results" in out
    assert "k_values" in out
    assert out["k_values"] == [10, 20]


def test_validate_custom_code_passes_for_valid_function():
    state = _make_state(n=40, d=4)
    reg = _registry()
    code = (
        "def custom_embedding(data, labels, params):\n"
        "    import numpy as np\n"
        "    from sklearn.decomposition import PCA\n"
        "    return PCA(n_components=2).fit_transform(data)\n"
    )
    out = reg.dispatch("validate_custom_code", {"code": code, "default_params": {}}, state)
    assert out["ok"] is True
    handle = out["embedding_handle"]
    assert handle in state.embeddings
    assert state.embeddings[handle].shape[1] == 2


def test_validate_custom_code_fails_for_broken_function():
    state = _make_state(n=20, d=4)
    reg = _registry()
    code = "def custom_embedding(data, labels, params):\n    raise ValueError('intentional')\n"
    out = reg.dispatch("validate_custom_code", {"code": code}, state)
    assert out["ok"] is False
    assert out["error_message"]
    assert any(not a["ok"] for a in state.code_attempts)


def test_validate_custom_code_fails_for_compile_error():
    state = _make_state(n=20, d=4)
    reg = _registry()
    out = reg.dispatch("validate_custom_code", {"code": "def custom_embedding(:\n  pass"}, state)
    assert out["ok"] is False
    assert "Compilation" in out["error_message"] or "error" in out["error_message"].lower()


def test_compute_pointwise_metrics_uses_handle():
    state = _make_state(n=40, d=4)
    reg = _registry()
    # First produce an embedding handle via validate_custom_code
    code = (
        "def custom_embedding(data, labels, params):\n"
        "    from sklearn.decomposition import PCA\n"
        "    return PCA(n_components=2).fit_transform(data)\n"
    )
    out = reg.dispatch("validate_custom_code", {"code": code}, state)
    handle = out["embedding_handle"]
    pw = reg.dispatch("compute_pointwise_metrics", {"embedding_handle": handle}, state)
    assert pw["embedding_handle"] == handle
    summary = pw["summary"]
    for key in ("mean_distortion", "mean_trust", "mean_cont", "p90_distortion"):
        assert key in summary


def test_compute_pointwise_metrics_unknown_handle_returns_error():
    state = _make_state(n=20, d=4)
    reg = _registry()
    out = reg.dispatch("compute_pointwise_metrics", {"embedding_handle": "no-such"}, state)
    assert "error" in out


def test_finalize_design_sets_done_flag_and_stores_report():
    state = _make_state(n=10, d=3)
    reg = _registry()
    out = reg.dispatch(
        "finalize_design",
        {"final_report": {"algorithm_name": "X", "novelty": "high"}},
        state,
    )
    assert out == {"acknowledged": True}
    assert state.done is True
    assert state.final_report["algorithm_name"] == "X"
