"""AllLLMAgenticPlanner — all-LLM design with the full agent toolset available.

Like the blank-canvas all_llm mode ("singleLLM"), but instead of a fixed phase
sequence, the LLM planner is handed the existing pipeline agents (1-6) as tools
and decides for itself whether and when to invoke each. Nothing is forced: no
mandatory data characterization (agent 1), no forced iterative optimization
(agent 6). The LLM writes a `custom_embedding` function, tests it on the full
dataset via the `test_algorithm` backend tool (execution + scoring stay fixed),
optionally consults any agent, iterates, and calls `finalize_design`.

The designed algorithm runs unsupervised when planner._unsupervised is True
(labels are withheld from the algorithm and used only for scoring); this is
enforced in NovelPlanner._exec_custom_code, reused by the test_algorithm tool.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .planner_novel import NovelPlanner
from .config import OUTPUT_ROOT
from .agent_loop import AgentLoop, SessionState
from .tools.registry import ToolRegistry
from .tools.compute_tools import register_compute_tools, register_finalize_tool
from .tools.agent_tools import register_agent_tools

logger = logging.getLogger(__name__)

_AFFINITY = ["euclidean_knn", "mutual_knn", "snn", "geodesic", "lof_weighted",
             "learnable", "perplexity_gaussian", "umap_fuzzy"]
_COST = ["kl_divergence", "cross_entropy", "spectral", "force_directed", "mutual_information"]
_LDK = ["student_t", "gaussian_ld", "umap_ab"]
_OPT = ["sgd", "adam", "spectral_eigen", "fft_accelerated"]


class AllLLMAgenticPlanner(NovelPlanner):
    """All-LLM planner that exposes agents 1-6 as tools and lets the LLM drive."""

    def __init__(self, config):
        super().__init__(config)
        self.mode_name = "all_llm_agentic"
        self.template_name = "DR_Template_v1"

    # ------------------------------------------------------------------
    def run(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_names: Optional[Dict[int, str]] = None,
        output_dir: str = OUTPUT_ROOT,
    ) -> Dict[str, Any]:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        n, d = data.shape

        self.context = {
            "mode": self.mode_name,
            "template_version": self.template_name,
            "_label_names": label_names,
            "data_spec": {
                "n_points": int(n),
                "original_dimension": int(d),
                "target_dimension": 2,
                "label_summary": self._label_summary(labels, label_names),
            },
            "design_constraints": {
                "primary_goal": self.config.primary_goal,
                "computational_budget": "moderate",
                "novelty_target": "unconstrained",
            },
            "available_methods": {"clustering_algorithms": ["kmeans", "agglomerative", "spectral"]},
            "available_primitives": {
                "affinity_primitives": _AFFINITY, "cost_primitives": _COST,
                "ld_kernel_primitives": _LDK, "optimization_primitives": _OPT,
            },
            "disabled_components": [],
        }

        # Deterministic, non-LLM precomputation so any agent the LLM *chooses* to
        # call has grounded inputs. This does NOT invoke agent 1 (characterization)
        # — only the PCA summary and baseline calibration the agents read from.
        self._precompute_data_summary(data)
        try:
            self._phase0_baseline_calibration(data, labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("baseline calibration failed: %s", exc)

        state = SessionState(
            data=data, labels=labels, label_names=label_names,
            primary_goal=self.config.primary_goal, output_dir=output_dir,
            dr_engine=self.dr_engine, metrics_engine=self.metrics_engine,
            clustering_engine=self.clustering_engine,
            metrics_k=self.config.metrics.neighborhood_k,
            lof_neighbors=self.config.metrics.lof_neighbors,
        )
        state.agent_outputs = {}
        state.baseline_profile = self.context.get("baseline_profile")

        registry = ToolRegistry()
        register_compute_tools(registry)     # deterministic helpers (pca, patches, quick validate…)
        register_agent_tools(registry, self)  # agents 1-6 + test_algorithm (full-data backend)
        register_finalize_tool(registry)

        loop = AgentLoop(
            llm_client=self.llm,
            planner_prompt=self._planner_prompt(),
            tool_registry=registry,
            session_state=state,
        )
        # Bound the loop: without this the LLM keeps refining up to 60 iterations,
        # which is impractical on large datasets where each test_algorithm runs a
        # full embedding. Override the class default per instance.
        loop.MAX_ITERATIONS = int(os.environ.get("DR_AGENTIC_MAX_ITERS", "16"))
        loop.add_initial_user_message(self._initial_message(n, d, labels, label_names))
        final_report = self._coerce_final_report(loop.run())
        final_report.setdefault("mode", self.mode_name)
        final_report["iteration_count"] = loop.iteration_count

        handle, embedding = self._resolve_best(state, final_report)
        if embedding is not None:
            np.save(os.path.join(output_dir, "best_embedding.npy"), embedding)
            metrics = state.metrics_cache.get(handle) or {}
            self.best_result = {"iteration": 0, "embedding": embedding,
                                "metrics": metrics, "handle": handle}
            algo_name = final_report.get("algorithm_name") or "AllLLM-Agentic Algorithm"
            self._novel_algo_name = algo_name
            try:
                full_metrics = self.metrics_engine.compute_all(
                    hd_data=data, ld_data=embedding, labels=labels,
                    k=self.config.metrics.neighborhood_k,
                    lof_neighbors=self.config.metrics.lof_neighbors)
                self._compare_with_baselines(
                    data=data, labels=labels, label_names=label_names,
                    novel_embedding=embedding, novel_metrics=full_metrics, algo_name=algo_name,
                    output_prefix="all_llm_agentic",
                    comparison_title=f"{algo_name} vs Baselines",
                    metrics_plot_title="Metrics Comparison — All-LLM-Agentic vs Baselines",
                    metrics_json_label=algo_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("baseline comparison failed: %s", exc)
        else:
            logger.warning("No usable embedding produced by the agentic loop.")

        final_report["tool_calls_summary"] = {
            "agent_outputs": sorted(state.agent_outputs.keys()),
            "n_test_algorithm_attempts": state.code_attempt_count,
            "chosen_handle": handle,
        }
        report_path = os.path.join(output_dir, "final_all_llm_agentic_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        logger.info("Final report saved → %s", report_path)
        return final_report

    # ------------------------------------------------------------------
    def _precompute_data_summary(self, data: np.ndarray) -> None:
        try:
            from sklearn.decomposition import PCA
            pca = PCA().fit(data)
            evr = pca.explained_variance_ratio_
            self.context["data_summary"] = json.dumps({
                "pca_explained_variance": {
                    "10": round(float(np.sum(evr[:10])), 4),
                    "20": round(float(np.sum(evr[:20])), 4),
                    "50": round(float(np.sum(evr[:min(50, data.shape[1])])), 4),
                }})
        except Exception as exc:  # noqa: BLE001
            logger.warning("data summary precompute failed: %s", exc)

    @staticmethod
    def _coerce_final_report(fr):
        """The finalize tool may hand back a JSON string or nest the payload under
        'final_report'. Flatten it so embedding_handle / algorithm_name survive."""
        if isinstance(fr, str):
            try:
                fr = json.loads(fr)
            except Exception:
                return {"raw": fr}
        if not isinstance(fr, dict):
            return {"raw": fr}
        out = dict(fr)
        inner = out.get("final_report")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = None
        if isinstance(inner, dict):
            out = {**inner, **{k: v for k, v in out.items() if k != "final_report"}}
        return out

    def _resolve_best(self, state, final_report):
        """Pick the final embedding:
        1. the handle the LLM explicitly finalized on (its deliberate choice);
        2. else, if a target metric is known, the tested embedding that scores
           best on that target (fair "best achieved", matching how the other
           modes report their optimized best);
        3. else the last successful test.
        """
        cand = final_report.get("embedding_handle")
        if isinstance(cand, str) and cand in state.embeddings:
            return cand, state.embeddings[cand]

        target = getattr(self, "_target_metric", None)  # (metric_key, +1/-1) or None
        if target and state.metrics_cache:
            key, sign = target
            best_h, best_score = None, None
            for h, m in state.metrics_cache.items():
                if h not in state.embeddings:
                    continue
                v = m.get(key)
                if not isinstance(v, (int, float)) or v != v:  # skip None/NaN
                    continue
                score = sign * float(v)
                if best_score is None or score > best_score:
                    best_h, best_score = h, score
            if best_h is not None:
                return best_h, state.embeddings[best_h]

        for atp in reversed(state.code_attempts):
            if atp.get("ok") and atp.get("handle") in state.embeddings:
                return atp["handle"], state.embeddings[atp["handle"]]
        handles = [h for h in state.embeddings if h.startswith("novel_attempt_")]
        if handles:
            h = max(handles, key=lambda x: int(x.rsplit("_", 1)[-1]))
            return h, state.embeddings[h]
        return None, None

    def _planner_prompt(self) -> str:
        return (
            "You are an autonomous algorithm designer for 2D dimensionality "
            "reduction. You design a NOVEL embedding algorithm from scratch and "
            "prove it out empirically.\n\n"
            "You have a set of specialist agent tools available. You decide freely "
            "whether, when, and in what order to use any of them — you may use all, "
            "some, or none:\n"
            "  - run_data_characterization (agent 1)\n"
            "  - run_difficulty_analysis (agent 2)\n"
            "  - run_neighborhood_analysis (agent 3)\n"
            "  - run_affinity_design (agent 4)\n"
            "  - run_cost_composition (agent 5)\n"
            "  - get_optimization_advice (agent 6)\n\n"
            "To evaluate a design, write a Python function "
            "`custom_embedding(data, labels, params)` returning an (n, 2) array "
            "(numpy/scipy/scikit-learn only; no torch) and call `test_algorithm` "
            "with the code. It runs on the FULL dataset and returns the full metric "
            "vector. IMPORTANT: the algorithm is UNSUPERVISED — `labels` will be "
            "None at runtime; design accordingly. Labels are used only to score.\n\n"
            "Iterate efficiently: you have a limited number of turns. Consult the "
            "agents you need up front, test a few strong candidate designs, and "
            "converge quickly. Do NOT keep testing marginal variants indefinitely. "
            "As soon as you have a design that clearly serves the primary goal, "
            "call `finalize_design` with a `final_report` that includes your best "
            "`embedding_handle`, an `algorithm_name`, a short `description`, and "
            "`key_innovations`. Aim to finalize within a handful of test_algorithm "
            "calls.\n\n"
            "Keep each algorithm reasonably efficient — avoid sweeping over dozens "
            "of random restarts on large datasets, as very slow code will time out. "
            "Optimize decisively for the stated primary goal."
        )

    def _initial_message(self, n, d, labels, label_names) -> str:
        prim = self.config.primary_goal
        ls = self._label_summary(labels, label_names) or "no labels available"
        return "\n".join([
            "# Design context",
            f"- n_points: {n}", f"- original_dimension: {d}", "- target_dimension: 2",
            f"- label_summary: {ls}",
            "",
            f"## PRIMARY GOAL\n{prim}",
            "",
            "Deterministic baselines (t-SNE / UMAP / PaCMAP) and a PCA summary have "
            "already been calibrated and are available to the agents you may call. "
            "Design your unsupervised algorithm, test it with test_algorithm, and "
            "finalize when your best result maximizes the primary goal.",
        ])
