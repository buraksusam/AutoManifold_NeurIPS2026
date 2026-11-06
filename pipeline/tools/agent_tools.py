"""Fixed-agent tools for the all-LLM agentic mode.

Exposes the existing pipeline agents (1-6) as tools the LLM planner may invoke
at will, plus a `test_algorithm` backend tool that runs an LLM-authored
`custom_embedding` on the FULL dataset (unsupervised-aware) and scores it.

Unlike the forced phase sequence in NovelPlanner, NOTHING here is invoked by the
harness. The LLM decides whether to characterize the data (agent 1), classify
difficulty (agent 2), analyze neighborhoods (agent 3), design affinities
(agent 4), compose a cost (agent 5), or ask for optimization advice (agent 6).

Each handler closes over the planner instance so it can reuse the planner's
agents, engines, deterministic precomputation, and `_exec_custom_code`
(which honors planner._unsupervised by passing labels=None into the algorithm).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

from .compute_tools import _flatten_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _merged_context(planner, state) -> Dict[str, Any]:
    """Base context (data_spec, design_constraints, primitives, baseline_profile,
    data_summary, patch_scores, grid_search_results) plus every agentN_output
    gathered so far this session."""
    ctx = dict(planner.context)
    ctx.update(getattr(state, "agent_outputs", {}) or {})
    return ctx


def _ensure_patch_scores(planner, state) -> None:
    if "patch_scores" not in planner.context:
        try:
            patch_scores, pd_, pr_ = planner._compute_patch_scores(state.data, state.labels)
            planner.context["patch_scores"] = json.dumps(patch_scores, indent=2, default=str)
            state.point_difficulty = pd_
            state.point_reliability = pr_
        except Exception as exc:  # noqa: BLE001
            logger.warning("patch score precompute failed: %s", exc)


def _ensure_grid(planner, state) -> None:
    if "grid_search_results" not in planner.context:
        try:
            grid = planner._compute_neighborhood_grid(state.data, state.labels)
            planner.context["grid_search_results"] = json.dumps(grid, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("neighborhood grid precompute failed: %s", exc)


# ---------------------------------------------------------------------------
# Agent adapter factory
# ---------------------------------------------------------------------------

def _make_agent_handler(planner, agent_attr, out_key, prep=None):
    def handler(args: Dict[str, Any], state) -> Dict[str, Any]:  # noqa: ARG001
        agent = getattr(planner, agent_attr, None)
        if agent is None:
            return {"error": f"agent '{agent_attr}' unavailable"}
        if prep is not None:
            prep(planner, state)
        ctx = _merged_context(planner, state)
        try:
            result = agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{agent_attr} failed: {exc}"}
        if not isinstance(result, dict):
            result = {"result": result}
        state.agent_outputs[out_key] = result
        try:
            planner._save_json(f"agentic_{out_key}.json", result)
        except Exception:  # noqa: BLE001
            pass
        return result
    return handler


def _make_optimization_advice_handler(planner):
    """Agent 6 — optimization advice for the current best/last-tested embedding."""
    def handler(args: Dict[str, Any], state) -> Dict[str, Any]:
        handle = args.get("embedding_handle") or getattr(state, "last_handle", None)
        if not handle or handle not in state.embeddings:
            return {"error": "no tested embedding available; call test_algorithm first"}
        metrics = state.metrics_cache.get(handle, {})
        ctx = _merged_context(planner, state)
        ctx.update({
            "current_iteration": len(state.code_attempts),
            "dr_method": "custom_novel",
            "current_hyperparameters": getattr(state, "last_params", {}) or {},
            "current_algorithm_code": getattr(state, "last_code", "") or "",
            "metrics": metrics,
            "embedding_summary": {"handle": handle,
                                  "shape": list(np.asarray(state.embeddings[handle]).shape)},
            "optimization_history": getattr(state, "opt_history", []),
        })
        agent = getattr(planner, "agent6", None)
        if agent is None:
            return {"error": "agent6 unavailable"}
        try:
            result = agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"agent6 failed: {exc}"}
        if not isinstance(result, dict):
            result = {"result": result}
        state.agent_outputs["agent6_output"] = result
        return result
    return handler


# ---------------------------------------------------------------------------
# Backend tool: full-data execution + scoring
# ---------------------------------------------------------------------------

def _make_test_algorithm_handler(planner):
    def handler(args: Dict[str, Any], state) -> Dict[str, Any]:
        code = args.get("code") or ""
        params = args.get("params") or {}
        if isinstance(params, str):  # Gemini sometimes serializes objects as strings
            try:
                params = json.loads(params) if params.strip() else {}
            except Exception:
                params = {}
        if not isinstance(params, dict):
            params = {}
        if not code.strip():
            return {"ok": False, "error_message": "empty code"}

        emb, ok, msg = planner._exec_custom_code(code, state.data, state.labels, params)
        if not ok:
            state.code_attempts.append({"ok": False, "code": code, "error_message": msg})
            return {"ok": False, "error_message": msg}

        state.code_attempt_count = getattr(state, "code_attempt_count", 0) + 1
        handle = f"novel_attempt_{state.code_attempt_count}"
        state.embeddings[handle] = emb

        flat: Dict[str, Any] = {}
        try:
            nested = planner.metrics_engine.compute_all(
                hd_data=state.data, ld_data=emb, labels=state.labels,
                k=state.metrics_k, lof_neighbors=state.lof_neighbors,
            )
            flat = _flatten_metrics(nested)
        except Exception as exc:  # noqa: BLE001
            flat = {"error": str(exc)}

        state.metrics_cache[handle] = flat
        state.code_attempts.append({"ok": True, "code": code, "handle": handle})
        state.last_handle = handle
        state.last_code = code
        state.last_params = params
        state.opt_history = getattr(state, "opt_history", [])
        state.opt_history.append({"handle": handle, "metrics": flat})
        return {
            "ok": True,
            "embedding_handle": handle,
            "embedding_shape": list(emb.shape),
            "metrics": flat,
            "note": "Metrics are on the FULL dataset. Labels were NOT passed to your "
                    "algorithm (unsupervised); they were used only to score the result.",
        }
    return handler


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_NOARG = {"type": "object", "properties": {}, "additionalProperties": False}

_AGENT_TOOLS = [
    ("run_data_characterization",
     "Agent 1 — characterize the dataset (intrinsic dimensionality, structure) "
     "and recommend preprocessing + candidate primitives. Optional; call if you "
     "want grounded data insight before designing.", _NOARG,
     lambda p: _make_agent_handler(p, "agent1", "agent1_output")),
    ("run_difficulty_analysis",
     "Agent 2 — classify regions of the data by embedding difficulty (Hard / "
     "Medium / Easy) and map each to suitable primitives.", _NOARG,
     lambda p: _make_agent_handler(p, "agent2", "agent2_output", prep=_ensure_patch_scores)),
    ("run_neighborhood_analysis",
     "Agent 3 — estimate maximal neighborhood sizes and give affinity-construction "
     "guidance from a neighborhood grid search.", _NOARG,
     lambda p: _make_agent_handler(p, "agent3", "agent3_output", prep=_ensure_grid)),
    ("run_affinity_design",
     "Agent 4 — design the high-dimensional affinity graph (primary primitive, "
     "hybrid blending, weighting) tailored to the goal and prior analysis.", _NOARG,
     lambda p: _make_agent_handler(p, "agent4", "agent4_output")),
    ("run_cost_composition",
     "Agent 5 — compose the cost/objective and low-dimensional kernel that matches "
     "the affinity design and the primary goal.", _NOARG,
     lambda p: _make_agent_handler(p, "agent5", "agent5_output")),
    ("get_optimization_advice",
     "Agent 6 — given your most recently tested embedding and its metrics, suggest "
     "hyperparameter / algorithm changes to improve the primary goal. Requires a "
     "prior test_algorithm call.",
     {"type": "object",
      "properties": {"embedding_handle": {"type": "string",
                     "description": "Handle to advise on; defaults to the last tested."}},
      "additionalProperties": False},
     _make_optimization_advice_handler),
]

_TEST_TOOL = (
    "test_algorithm",
    "Execute your `custom_embedding(data, labels, params)` function on the FULL "
    "dataset and score it. `labels` is passed as None (unsupervised); scoring uses "
    "the true labels. Returns the full metric vector. Call this to evaluate and "
    "iterate on your design.",
    {"type": "object",
     "properties": {
         "code": {"type": "string",
                  "description": "Python source defining custom_embedding(data, labels, params) "
                                 "-> np.ndarray of shape (n, 2). numpy/scipy/sklearn only; no torch."},
         "params": {"type": "object", "description": "Default parameters passed to your function."},
     },
     "required": ["code"], "additionalProperties": True},
)


def register_agent_tools(registry, planner) -> None:
    """Register agent adapter tools + the test_algorithm backend tool."""
    for name, desc, schema, factory in _AGENT_TOOLS:
        registry.register(name, {"description": desc, "input_schema": schema}, factory(planner))
    name, desc, schema = _TEST_TOOL
    registry.register(name, {"description": desc, "input_schema": schema},
                      _make_test_algorithm_handler(planner))
