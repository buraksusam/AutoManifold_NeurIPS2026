"""
Planner / Orchestrator — hyperparameter-optimization-only pipeline.

In ``optimization`` mode the pipeline is intentionally restricted to
tuning the hyperparameters of a fixed, well-known DR algorithm
(t-SNE / UMAP / PaCMAP / PCA). No custom algorithm code is generated;
algorithm-design phases and code-update suggestions from Agent 6 are
skipped. For novel-algorithm generation use ``planner_novel.py`` or
``planner_all_llm.py``.

Phase 1: Data Characterisation      (Agent 1)
Phase 2: Auxiliary Analysis          (Agents 2-5, sequential)
Phase 3: Baseline Setup              (record defaults — no LLM design)
Phase 4: Iterative Hyperparameter Optimisation (Agents 6, 7 in a loop)
Phase 5: Default-vs-Best Comparison  (DefaultComparisonAgent)
Phase 6: Visual Forensics            (Agent 9 — post-optimisation analysis)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .config import OUTPUT_ROOT, PipelineConfig
from .llm_client import LLMClient

# Compute engines
from .engines.dr_engine import DRVisualizationEngine
from .engines.metrics_engine import QuantitativeAnalystEngine
from .engines.clustering_engine import ClusteringEngine

# Visualisations
from .visualizations import (
    plot_data_overview,
    plot_difficulty_map,
    plot_outlier_overlay,
    plot_convergence_curve,
    plot_distortion_heatmap,
    plot_highlighted_areas,
    plot_class_distortion_summary,
    plot_dendrogram_comparison,
    plot_hierarchy_change_detail,
)

# Worker agents
from .agents.agent_1_data_characterization import DataCharacterizationAgent
from .agents.agent_2_difficulty_analyzer import EmbeddingDifficultyAgent
from .agents.agent_3_neighborhood_analyzer import NeighborhoodAnalyzerAgent
from .agents.agent_4_patch_stitching import PatchStitchingAgent
from .agents.agent_5_outlier_identification import OutlierIdentificationAgent
from .agents.agent_6_hyperparameter_opt import HyperparameterOptAgent
from .agents.agent_7_visualization_evaluator import VisualizationEvaluatorAgent
from .agents.agent_9_visual_forensics import VisualForensicsAgent
from .agents.agent_default_comparison import DefaultComparisonAgent

logger = logging.getLogger(__name__)


class Planner:
    """
    Central orchestrator implementing the 5-phase workflow of the paper's
    iterative pipeline (Figure 1).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = LLMClient(config.llm)

        # Compute engines
        self.dr_engine = DRVisualizationEngine()
        self.metrics_engine = QuantitativeAnalystEngine()
        self._metrics_k = config.metrics.neighborhood_k
        self._lof_neighbors = config.metrics.lof_neighbors
        self.clustering_engine = ClusteringEngine()

        # Worker agents (all share the same LLM client)
        self.agent1 = DataCharacterizationAgent(self.llm)
        self.agent2 = EmbeddingDifficultyAgent(self.llm)
        self.agent3 = NeighborhoodAnalyzerAgent(self.llm)
        self.agent4 = PatchStitchingAgent(self.llm)
        self.agent5 = OutlierIdentificationAgent(self.llm)
        self.agent6 = HyperparameterOptAgent(self.llm)
        self.agent7 = VisualizationEvaluatorAgent(self.llm)
        self.agent9 = VisualForensicsAgent(self.llm)
        self.agent_cmp = DefaultComparisonAgent(self.llm)

        # Pipeline state
        self.context: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.output_dir: str = OUTPUT_ROOT

    # ------------------------------------------------------------------
    # Terminal progress helper
    # ------------------------------------------------------------------

    @staticmethod
    def _stage(icon: str, phase: str, detail: str = "") -> None:
        """Print a clean one-line stage banner."""
        suffix = f"  › {detail}" if detail else ""
        print(f"\n{'─'*60}", flush=True)
        print(f"  {icon}  {phase}{suffix}", flush=True)
        print(f"{'─'*60}", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_names: Optional[Dict[int, str]] = None,
        output_dir: str = OUTPUT_ROOT,
    ) -> Dict[str, Any]:
        """Execute the full 5-phase pipeline on *data* (n × D)."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        n, d = data.shape
        self.context = {
            "_label_names": label_names,   # kept for visualisation helpers
            "data_spec": {
                "n_points": int(n),
                "original_dimension": int(d),
                "label_summary": self._label_summary(labels, label_names),
            },
            "design_constraints": {
                "primary_goal": self.config.primary_goal,
                "acceptable_minimum_quality_score": self.config.convergence.quality_score_target,
            },
            "available_methods": {
                "dr_algorithms": ["tsne", "umap", "pacmap", "pca"],
                "clustering_algorithms": ["kmeans", "agglomerative", "spectral"],
            },
        }

        self._stage("📊", "PHASE 1 — Data Characterisation")
        self._phase1_characterise(data, labels)

        self._stage("🔍", "PHASE 2 — Auxiliary Analysis", "Agents 2-5")
        self._phase2_auxiliary(data, labels)

        self._stage("🧰", "PHASE 3 — Baseline Setup", "hyperparameter-only mode")
        self._phase3_baseline_setup()

        self._stage("🔄", "PHASE 4 — Iterative Hyperparameter Optimisation")
        self._phase4_optimise(data, labels, label_names)

        self._stage("📐", "PHASE 5 — Default-vs-Best Comparison")
        final = self._phase5_compare_with_defaults(data, labels, label_names)

        self._stage("🔬", "PHASE 6 — Visual Forensics")
        vi_result = self._phase6_visual_forensics(data, labels, label_names)
        final["visual_forensics"] = vi_result

        # Persist final report
        report_path = os.path.join(output_dir, "final_report.json")
        with open(report_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final report saved → %s", report_path)

        return final

    # ------------------------------------------------------------------
    # Helper — persist any dict to a named JSON file in output_dir
    # ------------------------------------------------------------------

    def _save_json(self, filename: str, data: Any) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _phase1_characterise(self, data: np.ndarray, labels: Optional[np.ndarray]):
        """Run Agent 1 — Data Characterization."""
        from sklearn.decomposition import PCA

        print("  [1/1] Characterising data …", flush=True)
        pca_full = PCA().fit(data)
        explained_10 = float(np.sum(pca_full.explained_variance_ratio_[:10]))
        explained_20 = float(np.sum(pca_full.explained_variance_ratio_[:20]))
        explained_50 = float(np.sum(pca_full.explained_variance_ratio_[:min(50, data.shape[1])]))

        self.context["data_summary"] = json.dumps({
            "pca_explained_variance": {"10": round(explained_10, 4),
                                       "20": round(explained_20, 4),
                                       "50": round(explained_50, 4)},
            "feature_stats": {
                "mean_range": float(np.mean(np.ptp(data, axis=0))),
                "mean_std": float(np.mean(np.std(data, axis=0))),
            },
        })

        # Save the fitted PCA for the overview plot
        self.context["_pca_model"] = pca_full

        result = self.agent1.run(self.context)
        self.context["agent1_output"] = result
        p = self._save_json("agent1_data_characterization.json", result)
        print(f"  → Saved: {p}", flush=True)

        intrinsic_dim = result.get("data_properties", {}).get("intrinsic_dimensionality_estimate", "?")
        print(f"  → Intrinsic dim ≈ {intrinsic_dim}  |  PC10={explained_10:.1%}  PC20={explained_20:.1%}", flush=True)

        # Phase 1 visualisation
        p = plot_data_overview(data, labels, self.context.get("_label_names"), pca_full, self.output_dir)
        print(f"  → Saved: {p}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _phase2_auxiliary(self, data: np.ndarray, labels: Optional[np.ndarray]):
        """Run Agents 2-5 sequentially (2→3→4, 5 independent)."""
        # --- Agent 2: Difficulty classification ---
        print("  [1/4] Computing patch reliability scores (Agent 2) …", flush=True)
        patch_scores, point_difficulty, point_reliability = self._compute_patch_scores(data, labels)
        self.context["patch_scores"] = json.dumps(patch_scores, indent=2, default=str)

        result2 = self.agent2.run(self.context)
        self.context["agent2_output"] = result2
        p = self._save_json("agent2_difficulty_classification.json", result2)
        print(f"  → Saved: {p}", flush=True)
        d = result2.get("difficulty_classification", {}).get("groups", {})
        h_pct = d.get("H", {}).get("percentage_of_data", "?")
        e_pct = d.get("E", {}).get("percentage_of_data", "?")
        print(f"  → H/M/E classified  |  Hard={h_pct}%  Easy={e_pct}%", flush=True)

        # --- Agent 3: Maximal embeddable neighborhoods ---
        print("  [2/4] Estimating maximal embeddable neighbourhoods (Agent 3) …", flush=True)
        grid_results = self._compute_neighborhood_grid(data, labels)
        self.context["grid_search_results"] = json.dumps(grid_results, indent=2, default=str)

        result3 = self.agent3.run(self.context)
        self.context["agent3_output"] = result3
        p = self._save_json("agent3_maximal_neighborhoods.json", result3)
        print(f"  → Saved: {p}", flush=True)
        opt_k = result3.get("maximal_neighborhoods", {}).get("global_estimate", {}).get("optimal_k", "?")
        print(f"  → Optimal max neighbourhood k ≈ {opt_k}", flush=True)

        # --- Agent 4: Patch stitching ---
        print("  [3/4] Designing patch stitching algorithm (Agent 4) …", flush=True)
        result4 = self.agent4.run(self.context)
        self.context["agent4_output"] = result4
        p = self._save_json("agent4_patch_stitching.json", result4)
        print(f"  → Saved: {p}", flush=True)
        algo = result4.get("stitching_algorithm", {}).get("name", "?")
        print(f"  → Stitching algorithm: {algo}", flush=True)

        # --- Agent 5: Outlier identification (also reuses baseline embedding) ---
        print("  [4/4] Identifying impossible-to-embed points (Agent 5) …", flush=True)
        baseline = self._compute_baseline(data, labels)
        self.context["baseline_metrics"] = json.dumps(baseline.get("metrics", {}), indent=2, default=str)
        self.context["lof_stats"] = json.dumps(baseline.get("lof", {}), indent=2, default=str)

        result5 = self.agent5.run(self.context)
        self.context["agent5_output"] = result5
        p = self._save_json("agent5_outlier_identification.json", result5)
        print(f"  → Saved: {p}", flush=True)
        n_outliers = result5.get("identified_points", {}).get("count", "?")
        pct_outliers = result5.get("identified_points", {}).get("percentage_of_total", "?")
        print(f"  → Outliers identified: {n_outliers} ({pct_outliers}%)", flush=True)

        # Difficulty map — use the baseline (preferred-method) embedding
        dr_method = self.config.default_dr_method
        p = plot_difficulty_map(
            baseline["embedding"], point_difficulty, point_reliability,
            self.output_dir, method=dr_method,
        )
        print(f"  → Saved: {p}", flush=True)

        # Outlier overlay visualisation using per-point LOF scores
        lof_scores = baseline.get("lof_scores")
        if lof_scores is not None:
            p = plot_outlier_overlay(
                baseline["embedding"], lof_scores,
                labels, self.context.get("_label_names"), 1.5, self.output_dir
            )
            print(f"  → Saved: {p}", flush=True)

    # ------------------------------------------------------------------
    # Phase 3 — Baseline setup (no algorithm design in optimization mode)
    # ------------------------------------------------------------------

    def _phase3_baseline_setup(self):
        """Record the DR method and its library defaults for later comparison.

        In optimization mode the pipeline does not design a new algorithm;
        it only tunes hyperparameters of ``self.config.default_dr_method``.
        """
        dr_method = self.config.default_dr_method
        self.context["dr_method"] = dr_method
        defaults = self._initial_hyperparams(dr_method)
        self.context["initial_hyperparameters"] = copy.deepcopy(defaults)
        print(f"  → Algorithm: {dr_method}", flush=True)
        print(f"  → Initial (default) hyperparameters: {defaults}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4 — Iterative Optimisation
    # ------------------------------------------------------------------

    def _phase4_optimise(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ):
        """Paper's iterative loop: compute → evaluate → adjust → repeat.

        Hyperparameter-only: always runs the standard DR engine for
        ``self.config.default_dr_method``. Any ``code_update`` field from
        Agent 6 is ignored — this mode tunes parameters only.
        """
        cfg = self.config
        conv = cfg.convergence
        dr_method = cfg.default_dr_method

        hyperparams = self._initial_hyperparams(dr_method)
        if not isinstance(hyperparams, dict):
            logger.warning("Initial hyperparameters were %s, resetting to {}", type(hyperparams).__name__)
            hyperparams = {}

        print(f"  ℹ️  Using standard {dr_method} engine (hyperparameter-only optimisation)", flush=True)

        # When iterative optimization is disabled, cap to a single pass.
        # Optimization mode IS hyperparameter tuning, so disabling it just means
        # "run defaults once and report metrics".
        effective_max_iter = conv.max_iterations
        if not conv.enable_iterative_optimization:
            effective_max_iter = 1
            print("  ℹ️  Iterative optimization disabled — capping to 1 iteration", flush=True)

        best_score = -1.0
        patience_counter = 0

        for iteration in range(1, effective_max_iter + 1):
            print(f"\n  [{iteration}/{effective_max_iter}] Computing embedding …", flush=True)

            # --- 1. Compute embedding (standard engine, current hyperparams) ---
            embedding = self.dr_engine.compute_embedding(
                data, method=dr_method, **hyperparams
            )

            # --- 2. Compute metrics ---
            metrics = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self._metrics_k, lof_neighbors=self._lof_neighbors,
            )

            # --- 3. Compute dendrograms ---
            dendrograms = self.clustering_engine.compute_dendrograms(
                hd_data=data, ld_data=embedding, labels=labels
            )

            # --- 3b. Compute hierarchy diff (HD vs 2D) ---
            hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
                dendrograms, label_names=label_names, dr_method=dr_method,
            )
            rf_dist = hierarchy_diff.get("robinson_foulds_distance", "?")
            coph_corr = hierarchy_diff.get("cophenetic_correlation", "?")
            n_topo_changes = len(hierarchy_diff.get("topology_changes", []))
            print(
                f"  → Hierarchy: RF={rf_dist}, coph_corr={coph_corr}, "
                f"topology_changes={n_topo_changes}",
                flush=True,
            )

            # --- 4. Render plot ---
            plot_path = os.path.join(
                self.output_dir, f"iter_{iteration:03d}.png"
            )

            # --- 5. Build iteration context ---
            algo_label = dr_method
            iter_ctx = {
                **self.context,
                "current_iteration": iteration,
                "dr_method": dr_method,
                "current_hyperparameters": hyperparams,
                "current_algorithm_code": "# Standard DR engine (hyperparameter-only mode)",
                "metrics": metrics,
                "dendrograms": dendrograms,
                "hierarchy_diff": hierarchy_diff,
                "embedding_summary": {
                    "shape": list(embedding.shape),
                    "range_x": [float(embedding[:, 0].min()), float(embedding[:, 0].max())],
                    "range_y": [float(embedding[:, 1].min()), float(embedding[:, 1].max())],
                },
                "optimization_history": self.optimization_history,
                "plot_path": plot_path,
            }

            # --- 6. Agent 7 — Independent evaluation ---
            eval_result = self.agent7.run(iter_ctx)
            quality_score = float(eval_result.get("overall_assessment", {}).get("quality_score", 0))
            self._save_json(f"iter_{iteration:03d}_agent7_evaluation.json", eval_result)

            # --- 7. Agent 6 — Hyperparameter optimization ---
            #     (code_update suggestions are ignored in optimization mode)
            opt_result = self.agent6.run(iter_ctx)
            if isinstance(opt_result, dict) and opt_result.get("code_update"):
                logger.info(
                    "Ignoring code_update from Agent 6 (optimization mode is "
                    "hyperparameter-only)"
                )
                opt_result["code_update"] = None
            implicit_score = float(opt_result.get("quality_score", 0))
            self._save_json(f"iter_{iteration:03d}_agent6_optimization.json", opt_result)

            # Implicit score drives convergence; evaluator is logged for comparison only
            convergence_score = implicit_score

            # --- 8. Render plot now that we have scores ---
            self.dr_engine.render_plot(
                embedding, labels=labels, label_names=label_names,
                title=f"{algo_label.upper()} — Iteration {iteration}",
                quality_score=implicit_score,
                save_path=plot_path,
            )

            # --- 9. Record history ---
            global_m = metrics.get("Global Structure & HD<->2D Correlation", {})
            local_m  = metrics.get("Neighborhood Preservation", {})
            cluster_m = metrics.get("Clustering & Label Agreement", {})
            lof_m    = metrics.get("Outlier Detection", {})
            record = {
                "iteration": iteration,
                "algorithm": algo_label,
                "hyperparameters": copy.deepcopy(hyperparams),
                "implicit_score": implicit_score,
                "evaluator_score": quality_score,
                "convergence_score": convergence_score,
                "code_updated": False,
                "metrics_snapshot": {
                    "Spearman": global_m.get("Spearman Correlation"),
                    "Stress": global_m.get("Stress"),
                    "Mean_Distance_Ratio": global_m.get("Mean Distance Ratio"),
                    "Trustworthiness": next((v for k, v in local_m.items() if k.startswith("Trustworthiness")), None),
                    "Continuity": next((v for k, v in local_m.items() if k.startswith("Continuity")), None),
                    "Silhouette": cluster_m.get("Silhouette Score"),
                    "LOF_Median": lof_m.get("LOF Median"),
                    "LOF_Outlier_pct": lof_m.get("LOF Outlier %"),
                },
                "agent6_score_rationale": (opt_result or {}).get("score_rationale", ""),
                "agent6_all_recommendations": (opt_result or {}).get("recommendations", []),
                "agent6_key_weaknesses": (opt_result or {}).get("overall_assessment", {}).get("key_weaknesses", []),
                "agent7_key_strengths": (eval_result or {}).get("overall_assessment", {}).get("key_strengths", []),
                "agent7_key_weaknesses": (eval_result or {}).get("overall_assessment", {}).get("key_weaknesses", []),
                "agent7_improvement_priorities": (eval_result or {}).get("improvement_priorities", [])[:3],
                "hierarchy_rf_distance": hierarchy_diff.get("robinson_foulds_distance"),
                "hierarchy_cophenetic_corr": hierarchy_diff.get("cophenetic_correlation"),
                "hierarchy_n_topology_changes": len(hierarchy_diff.get("topology_changes", [])),
            }
            self.optimization_history.append(record)
            self.context["optimization_history"] = self.optimization_history

            # Write per-iteration metrics immediately so the UI can pick them up
            iter_metrics_path = os.path.join(self.output_dir, f"iter_{iteration}_metrics.json")
            with open(iter_metrics_path, "w") as _f:
                json.dump({
                    "iteration": iteration,
                    "metrics": record["metrics_snapshot"],
                    "implicit_score": record["implicit_score"],
                    "evaluator_score": record["evaluator_score"],
                }, _f, indent=2, default=str)

            print(
                f"  → implicit={implicit_score:.2f} (convergence)  "
                f"evaluator={quality_score:.2f} (reference)  "
                f"best={best_score:.2f}",
                flush=True,
            )

            # --- 10. Check convergence ---
            if convergence_score >= conv.quality_score_target:
                print(f"  ✅ Target quality {conv.quality_score_target} reached — converging.", flush=True)
                self.best_result = {
                    "iteration": iteration,
                    "hyperparameters": copy.deepcopy(hyperparams),
                    "embedding": embedding,
                    "metrics": metrics,
                    "dendrograms": dendrograms,
                    "eval_result": eval_result,
                    "opt_result": opt_result,
                    "plot_path": plot_path,
                }
                break

            if convergence_score > best_score:
                best_score = convergence_score
                patience_counter = 0
                self.best_result = {
                    "iteration": iteration,
                    "hyperparameters": copy.deepcopy(hyperparams),
                    "embedding": embedding,
                    "metrics": metrics,
                    "dendrograms": dendrograms,
                    "eval_result": eval_result,
                    "opt_result": opt_result,
                    "plot_path": plot_path,
                }
            else:
                patience_counter += 1

            if patience_counter >= conv.patience:
                print(f"  ⏹ No improvement for {conv.patience} iterations — stopping early.", flush=True)
                break

            # --- 11. Apply hyperparameter recommendations ---
            old_params = copy.deepcopy(hyperparams)
            hyperparams, applied_recs, skipped_recs = self._apply_recommendations(
                hyperparams, opt_result
            )
            if not isinstance(hyperparams, dict):
                logger.warning(
                    "Optimizer returned non-dict hyperparameters (%s); keeping previous params",
                    type(hyperparams).__name__,
                )
                hyperparams = copy.deepcopy(old_params)

            # Compute what actually changed
            params_delta = {}
            all_keys = set(list(old_params.keys()) + list(hyperparams.keys()))
            for k in all_keys:
                if old_params.get(k) != hyperparams.get(k):
                    params_delta[k] = {"from": old_params.get(k), "to": hyperparams.get(k)}

            # Annotate the latest history record with what was actually applied
            if self.optimization_history:
                self.optimization_history[-1]["params_delta"] = params_delta
                self.optimization_history[-1]["applied_recommendations"] = applied_recs
                self.optimization_history[-1]["skipped_recommendations"] = skipped_recs

            # Stale iteration — no parameter changes
            if not params_delta:
                print(
                    f"  ⚠️  WARNING: No parameter changes this iteration! "
                    f"Agent 6 will be informed to try something different.",
                    flush=True,
                )
                if self.optimization_history:
                    self.optimization_history[-1]["stale_iteration"] = True
            else:
                changes_str = ", ".join(f"{k}: {v['from']}→{v['to']}" for k, v in params_delta.items())
                print(f"  → Changes: {changes_str}", flush=True)
            print(f"  → Updated params: {hyperparams}", flush=True)

        # Save iteration history
        hist_path = os.path.join(self.output_dir, "optimization_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.optimization_history, f, indent=2, default=str)

        # Convergence visualisation
        if self.optimization_history:
            p = plot_convergence_curve(
                self.optimization_history,
                self.output_dir,
                quality_target=self.config.convergence.quality_score_target,
            )
            print(f"  → Convergence plot saved: {p}", flush=True)

    # ------------------------------------------------------------------
    # Phase 5 — Default-vs-Best comparison
    # ------------------------------------------------------------------

    def _phase5_compare_with_defaults(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        """Compute an embedding with the algorithm's default hyperparameters
        and compare its metrics with the optimized best result.
        """
        dr_method = self.config.default_dr_method

        if not self.best_result:
            print("  ⚠️  No best result available — skipping comparison.", flush=True)
            return {
                "best_result": None,
                "default_vs_best_comparison": None,
                "optimization_history": self.optimization_history,
            }

        default_params = self._initial_hyperparams(dr_method)
        best_params = self.best_result.get("hyperparameters", {}) or {}
        best_metrics = self.best_result.get("metrics", {}) or {}
        best_snap = self._extract_metrics_snapshot(best_metrics)

        print(f"  [1/3] Computing baseline embedding with default {dr_method} params …", flush=True)
        default_embedding = self.dr_engine.compute_embedding(
            data, method=dr_method, **default_params
        )
        default_metrics = self.metrics_engine.compute_all(
            hd_data=data, ld_data=default_embedding, labels=labels,
            k=self._metrics_k, lof_neighbors=self._lof_neighbors,
        )
        default_snap = self._extract_metrics_snapshot(default_metrics)

        # Parameter diff
        all_keys = set(list(default_params.keys()) + list(best_params.keys()))
        param_diff = {
            k: {"default": default_params.get(k), "best": best_params.get(k)}
            for k in sorted(all_keys)
            if default_params.get(k) != best_params.get(k)
        }

        # Quantitative metric deltas
        METRIC_DIRS = {
            "Trustworthiness": "higher_better",
            "Continuity": "higher_better",
            "Spearman": "higher_better",
            "Silhouette": "higher_better",
            "Stress": "lower_better",
            "LOF_Median": "lower_better",
        }
        metric_deltas = []
        for metric, direction in METRIC_DIRS.items():
            d_val = default_snap.get(metric)
            b_val = best_snap.get(metric)
            if d_val is None or b_val is None:
                delta = None
                verdict = "unavailable"
            else:
                delta = float(b_val) - float(d_val)
                if abs(delta) < 0.005:
                    verdict = "unchanged"
                elif direction == "higher_better":
                    verdict = "better" if delta > 0 else "worse"
                else:
                    verdict = "better" if delta < 0 else "worse"
            metric_deltas.append({
                "metric": metric,
                "default_value": None if d_val is None else float(d_val),
                "best_value": None if b_val is None else float(b_val),
                "delta": None if delta is None else float(delta),
                "direction": direction,
                "verdict": verdict,
            })

        # --- 2. LLM commentary via the comparison agent ---
        print(f"  [2/3] Asking DefaultComparisonAgent for verdict …", flush=True)
        agent_ctx = {
            "dr_method": dr_method,
            "default_params": default_params,
            "best_params": best_params,
            "param_diff": param_diff,
            "default_metrics_snapshot": default_snap,
            "best_metrics_snapshot": best_snap,
            "primary_goal": self.config.primary_goal,
        }
        llm_result = self.agent_cmp.run(agent_ctx)

        # --- 3. Render side-by-side comparison plot ---
        print(f"  [3/3] Rendering side-by-side default-vs-best plot …", flush=True)
        default_plot_path = os.path.join(self.output_dir, "default_params_embedding.png")
        self.dr_engine.render_plot(
            default_embedding, labels=labels, label_names=label_names,
            title=f"{dr_method.upper()} — Default Parameters",
            save_path=default_plot_path,
        )
        comparison_plot_path = self._plot_default_vs_best(
            default_embedding=default_embedding,
            best_embedding=self.best_result.get("embedding"),
            default_snap=default_snap,
            best_snap=best_snap,
            labels=labels,
            label_names=label_names,
            dr_method=dr_method,
        )

        comparison_report = {
            "algorithm": dr_method,
            "default_params": default_params,
            "best_params": best_params,
            "param_diff": param_diff,
            "default_metrics_snapshot": default_snap,
            "best_metrics_snapshot": best_snap,
            "metric_deltas": metric_deltas,
            "llm_verdict": llm_result,
            "plots": {
                "default_embedding": default_plot_path,
                "comparison_side_by_side": comparison_plot_path,
            },
        }
        self._save_json("default_vs_best_comparison.json", comparison_report)
        print(
            f"  → Verdict: {llm_result.get('overall_verdict', 'unknown')} "
            f"(recommend adopt optimized = {llm_result.get('recommend_adopt_optimized_params', '?')})",
            flush=True,
        )

        return {
            "best_result": {
                "iteration": self.best_result.get("iteration"),
                "hyperparameters": self.best_result.get("hyperparameters"),
                "metrics": self.best_result.get("metrics"),
                "plot_path": self.best_result.get("plot_path"),
            },
            "default_vs_best_comparison": comparison_report,
            "optimization_history": self.optimization_history,
        }

    # ------------------------------------------------------------------
    # Helpers — metrics snapshot & comparison plot
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics_snapshot(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metrics into a simple snapshot dict."""
        global_m = metrics.get("Global Structure & HD<->2D Correlation", {})
        local_m = metrics.get("Neighborhood Preservation", {})
        cluster_m = metrics.get("Clustering & Label Agreement", {})
        lof_m = metrics.get("Outlier Detection", {})
        return {
            "Spearman": global_m.get("Spearman Correlation"),
            "Stress": global_m.get("Stress"),
            "Mean_Distance_Ratio": global_m.get("Mean Distance Ratio"),
            "Trustworthiness": next(
                (v for k, v in local_m.items() if k.startswith("Trustworthiness")), None
            ),
            "Continuity": next(
                (v for k, v in local_m.items() if k.startswith("Continuity")), None
            ),
            "Silhouette": cluster_m.get("Silhouette Score"),
            "LOF_Median": lof_m.get("LOF Median"),
            "LOF_Outlier_pct": lof_m.get("LOF Outlier %"),
        }

    def _plot_default_vs_best(
        self,
        default_embedding: np.ndarray,
        best_embedding: Optional[np.ndarray],
        default_snap: Dict[str, Any],
        best_snap: Dict[str, Any],
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        dr_method: str,
    ) -> Optional[str]:
        """Side-by-side scatter of default vs best embedding with metric sub-titles."""
        if best_embedding is None:
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        palette = [
            "#E74C3C", "#2E86AB", "#27AE60", "#F39C12", "#8E44AD",
            "#16A085", "#D35400", "#2C3E50", "#C0392B", "#1ABC9C",
        ]

        def _scatter(ax, emb, title_str, subtitle):
            if labels is not None:
                for i, lab in enumerate(np.unique(labels)):
                    mask = labels == lab
                    name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
                    ax.scatter(emb[mask, 0], emb[mask, 1], c=palette[i % len(palette)],
                               label=name, s=6, alpha=0.7, edgecolors="none")
                ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c="#2E86AB", edgecolors="none")
            ax.set_title(title_str, fontsize=11, fontweight="bold")
            if subtitle:
                ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, fontsize=8,
                        ha="center", color="#555")
            ax.set_xticks([]); ax.set_yticks([])

        def _subtitle(snap):
            parts = []
            for k in ("Trustworthiness", "Continuity", "Stress", "Silhouette"):
                v = snap.get(k)
                if v is not None:
                    parts.append(f"{k[:4]}={float(v):.3f}")
            return "  ".join(parts)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        _scatter(axes[0], default_embedding,
                 f"{dr_method.upper()} — Default Params", _subtitle(default_snap))
        _scatter(axes[1], best_embedding,
                 f"★ {dr_method.upper()} — Best (Optimized) Params", _subtitle(best_snap))
        fig.suptitle(
            f"Hyperparameter Optimization — Default vs Best ({dr_method.upper()})",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        path = os.path.join(self.output_dir, "default_vs_best_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Comparison plot saved: {path}", flush=True)
        return path

    # ------------------------------------------------------------------
    # Phase 6 — Visual Forensics
    # ------------------------------------------------------------------

    def _phase6_visual_forensics(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        """
        Post-optimisation visual forensics analysis:
        1. Compute pointwise distortion metrics on the best embedding
        2. Cluster high-distortion regions for the agent
        3. Agent 9 analyses everything and outputs region coordinates
        4. Generate annotated plots with distortion heatmaps + highlighted areas
        """
        if not self.best_result:
            print("  ⚠️  No best result available — skipping visual forensics.", flush=True)
            return {}

        embedding = self.best_result["embedding"]
        best_metrics = self.best_result.get("metrics", {})
        n_points = embedding.shape[0]

        print(f"  Computing pointwise distortion metrics for {n_points} points …", flush=True)

        # --- 1. Compute pointwise metrics ---
        pw_metrics = self.metrics_engine.compute_pointwise_metrics(
            hd_data=data, ld_data=embedding, k=self._metrics_k,
        )
        pw_trust = pw_metrics["pointwise_trust"]
        pw_cont = pw_metrics["pointwise_cont"]
        pw_stress = pw_metrics["pointwise_stress"]
        composite = pw_metrics["composite_distortion"]

        print(
            f"  → Distortion stats: mean={composite.mean():.4f}, "
            f"median={np.median(composite):.4f}, "
            f"p90={np.percentile(composite, 90):.4f}, "
            f"p95={np.percentile(composite, 95):.4f}",
            flush=True,
        )

        # --- 2. Identify high-distortion clusters ---
        # Find spatial clusters of high-distortion points using a simple approach:
        # take the worst 10% by composite distortion, cluster them spatially
        distortion_clusters = self._find_distortion_clusters(
            embedding, composite, threshold_pct=90,
        )
        print(f"  → Found {len(distortion_clusters)} high-distortion clusters", flush=True)

        # --- 3. Compute per-class distortion stats ---
        class_centroids = {}
        class_distortion_stats = {}
        if labels is not None:
            for lab in np.unique(labels):
                mask = labels == lab
                name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
                cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
                class_centroids[name] = {"x": round(float(cx), 2), "y": round(float(cy), 2)}
                class_distortion_stats[name] = {
                    "n_points": int(mask.sum()),
                    "mean_distortion": round(float(composite[mask].mean()), 4),
                    "mean_trust": round(float(pw_trust[mask].mean()), 4),
                    "mean_cont": round(float(pw_cont[mask].mean()), 4),
                    "p90_distortion": round(float(np.percentile(composite[mask], 90)), 4),
                    "pct_high_distortion": round(
                        float((composite[mask] > np.percentile(composite, 90)).sum() / mask.sum() * 100), 1
                    ),
                }

        # --- 4. Compute worst-region centroids ---
        worst_10_mask = composite >= np.percentile(composite, 90)
        worst_trust_mask = pw_trust <= np.percentile(pw_trust, 10)
        worst_cont_mask = pw_cont <= np.percentile(pw_cont, 10)

        def _centroid(mask):
            if mask.any():
                return float(embedding[mask, 0].mean()), float(embedding[mask, 1].mean())
            return 0.0, 0.0

        worst_dist_cx, worst_dist_cy = _centroid(worst_10_mask)
        worst_trust_cx, worst_trust_cy = _centroid(worst_trust_mask)
        worst_cont_cx, worst_cont_cy = _centroid(worst_cont_mask)

        # --- 5. Compute hierarchy diff for forensics ---
        print("  Computing hierarchy comparison (HD vs 2D) …", flush=True)
        best_dendrograms = self.best_result.get("dendrograms", {})
        # If dendrograms don't have linkage data (old format), recompute
        if best_dendrograms.get("hierarchy_hd", {}).get("linkage") is None:
            best_dendrograms = self.clustering_engine.compute_dendrograms(
                hd_data=data, ld_data=embedding, labels=labels,
            )
        hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
            best_dendrograms, label_names=label_names,
            dr_method=self.config.default_dr_method,
        )
        rf_dist = hierarchy_diff.get("robinson_foulds_distance", "?")
        coph_corr = hierarchy_diff.get("cophenetic_correlation", "?")
        n_topo = len(hierarchy_diff.get("topology_changes", []))
        print(
            f"  → Hierarchy: RF={rf_dist}, cophenetic_corr={coph_corr}, "
            f"topology_changes={n_topo}",
            flush=True,
        )

        # --- 6. Build agent context ---
        agent_ctx = {
            "embedding_summary": {
                "shape": list(embedding.shape),
                "ranges": {
                    "x_min": float(embedding[:, 0].min()),
                    "x_max": float(embedding[:, 0].max()),
                    "y_min": float(embedding[:, 1].min()),
                    "y_max": float(embedding[:, 1].max()),
                },
            },
            "pointwise_metrics": {
                "mean_distortion": float(composite.mean()),
                "median_distortion": float(np.median(composite)),
                "std_distortion": float(composite.std()),
                "p90_distortion": float(np.percentile(composite, 90)),
                "p95_distortion": float(np.percentile(composite, 95)),
                "mean_trust": float(pw_trust.mean()),
                "mean_cont": float(pw_cont.mean()),
                "worst_trust_cx": worst_trust_cx,
                "worst_trust_cy": worst_trust_cy,
                "worst_cont_cx": worst_cont_cx,
                "worst_cont_cy": worst_cont_cy,
                "worst_dist_cx": worst_dist_cx,
                "worst_dist_cy": worst_dist_cy,
            },
            "class_centroids": class_centroids,
            "class_distortion_stats": class_distortion_stats,
            "distortion_clusters": distortion_clusters,
            "hierarchy_diff": hierarchy_diff,
            "best_result": self.best_result,
            "optimization_history": self.optimization_history,
        }

        # --- 7. Run Agent 9 ---
        print("  Running Agent 9 — Visual Forensics Evaluator …", flush=True)
        vi_result = self.agent9.run(agent_ctx)
        p = self._save_json("agent9_visual_forensics.json", vi_result)
        print(f"  → Saved: {p}", flush=True)

        # --- 8. Generate plots ---
        print("  Generating distortion heatmap …", flush=True)
        p1 = plot_distortion_heatmap(
            embedding=embedding,
            composite_distortion=composite,
            pointwise_trust=pw_trust,
            pointwise_cont=pw_cont,
            pointwise_stress=pw_stress,
            labels=labels,
            label_names=label_names,
            output_dir=self.output_dir,
        )
        print(f"  → Saved: {p1}", flush=True)

        # Extract highlighted regions from agent output
        highlighted_regions = vi_result.get("highlighted_regions", [])
        if highlighted_regions:
            print(f"  Generating highlighted areas plot ({len(highlighted_regions)} regions) …", flush=True)
            p2 = plot_highlighted_areas(
                embedding=embedding,
                labels=labels,
                label_names=label_names,
                highlighted_regions=highlighted_regions,
                composite_distortion=composite,
                output_dir=self.output_dir,
            )
            print(f"  → Saved: {p2}", flush=True)
        else:
            print("  ⚠️  Agent 9 returned no highlighted regions.", flush=True)
            p2 = None

        # Per-class distortion summary
        if labels is not None:
            print("  Generating per-class distortion summary …", flush=True)
            p3 = plot_class_distortion_summary(
                embedding=embedding,
                composite_distortion=composite,
                pointwise_trust=pw_trust,
                pointwise_cont=pw_cont,
                labels=labels,
                label_names=label_names,
                output_dir=self.output_dir,
            )
            print(f"  → Saved: {p3}", flush=True)
        else:
            p3 = None

        # --- 8b. Dendrogram comparison plots ---
        print("  Generating dendrogram comparison plots …", flush=True)
        p4 = plot_dendrogram_comparison(
            dendrograms=best_dendrograms,
            hierarchy_diff=hierarchy_diff,
            label_names=label_names,
            output_dir=self.output_dir,
        )
        if p4:
            print(f"  → Saved: {p4}", flush=True)

        # Hierarchy change detail (arrows on 2D scatter + merge-rank chart)
        p5 = plot_hierarchy_change_detail(
            dendrograms=best_dendrograms,
            hierarchy_diff=hierarchy_diff,
            embedding=embedding,
            labels=labels,
            label_names=label_names,
            output_dir=self.output_dir,
        )
        if p5:
            print(f"  → Saved: {p5}", flush=True)

        return {
            "agent9_output": vi_result,
            "plots": {
                "distortion_heatmap": p1,
                "highlighted_areas": p2,
                "class_distortion_summary": p3,
                "dendrogram_comparison": p4,
                "hierarchy_change_detail": p5,
            },
            "pointwise_stats": {
                "mean_distortion": float(composite.mean()),
                "p90_distortion": float(np.percentile(composite, 90)),
                "mean_trust": float(pw_trust.mean()),
                "mean_cont": float(pw_cont.mean()),
            },
            "hierarchy_diff": {
                "robinson_foulds_distance": hierarchy_diff.get("robinson_foulds_distance"),
                "cophenetic_correlation": hierarchy_diff.get("cophenetic_correlation"),
                "n_topology_changes": len(hierarchy_diff.get("topology_changes", [])),
                "topology_changes": hierarchy_diff.get("topology_changes", []),
            },
            "n_highlighted_regions": len(highlighted_regions),
            "distortion_clusters": distortion_clusters,
            "class_distortion_stats": class_distortion_stats,
        }

    def _find_distortion_clusters(
        self,
        embedding: np.ndarray,
        composite_distortion: np.ndarray,
        threshold_pct: float = 90,
        max_clusters: int = 8,
    ) -> list:
        """
        Identify spatial clusters of high-distortion points.
        Uses simple DBSCAN on the worst-percentile points in 2D space.
        Returns a list of dicts with center_x, center_y, radius, n_points, mean_distortion.
        """
        from sklearn.cluster import DBSCAN

        threshold = np.percentile(composite_distortion, threshold_pct)
        high_mask = composite_distortion >= threshold
        if high_mask.sum() < 3:
            return []

        high_points = embedding[high_mask]
        high_dist = composite_distortion[high_mask]

        # Set eps relative to the spread of the embedding
        x_range = embedding[:, 0].max() - embedding[:, 0].min()
        y_range = embedding[:, 1].max() - embedding[:, 1].min()
        eps = max(x_range, y_range) * 0.08  # 8% of extent

        db = DBSCAN(eps=eps, min_samples=max(3, int(high_mask.sum() * 0.05)))
        cluster_labels = db.fit_predict(high_points)

        clusters = []
        for cl in sorted(set(cluster_labels)):
            if cl == -1:
                continue  # skip noise
            cmask = cluster_labels == cl
            pts = high_points[cmask]
            dists = high_dist[cmask]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            # Radius = max distance from centroid + small margin
            distances_to_center = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            radius = float(np.percentile(distances_to_center, 85) + eps * 0.3)

            clusters.append({
                "cluster_id": int(cl),
                "center_x": round(float(cx), 2),
                "center_y": round(float(cy), 2),
                "radius": round(radius, 2),
                "n_points": int(cmask.sum()),
                "mean_distortion": round(float(dists.mean()), 4),
                "max_distortion": round(float(dists.max()), 4),
            })

        # Sort by mean distortion (worst first) and limit
        clusters.sort(key=lambda c: -c["mean_distortion"])
        return clusters[:max_clusters]

    # ------------------------------------------------------------------
    # Helpers — pre-computation for agents
    # ------------------------------------------------------------------

    def _compute_patch_scores(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> tuple:
        """Cluster data into ~100-point patches, embed each, score.

        Returns
        -------
        results : list of dicts  (patch metrics, one per patch)
        point_difficulty : np.ndarray of shape (n,) with values 'H'/'M'/'E'
        point_reliability : np.ndarray of shape (n,) floats 0-1
        """
        from sklearn.cluster import MiniBatchKMeans

        n = data.shape[0]
        target_size = self.config.patch.target_patch_size
        n_patches = max(1, n // target_size)

        km = MiniBatchKMeans(n_clusters=n_patches, random_state=42, batch_size=min(1024, n))
        patch_labels = km.fit_predict(data)

        results = []
        patch_reliability_scores: Dict[int, float] = {}
        for pid in range(n_patches):
            mask = patch_labels == pid
            patch_data = data[mask]
            if patch_data.shape[0] < 5:
                continue
            patch_lbl = labels[mask] if labels is not None else None

            scores: Dict[str, Any] = {"patch_id": pid, "size": int(patch_data.shape[0])}
            rel_vals = []
            for method in ["tsne", "umap"]:
                try:
                    emb = self.dr_engine.compute_embedding(patch_data, method=method, n_pcs=min(20, patch_data.shape[1]))
                    m = self.metrics_engine.compute_all(hd_data=patch_data, ld_data=emb, labels=patch_lbl,
                                                         k=self._metrics_k,
                                                         lof_neighbors=self._lof_neighbors)
                    # Metrics are nested — traverse to the correct keys
                    local_m = m.get("Neighborhood Preservation", {})
                    global_m = m.get("Global Structure & HD<->2D Correlation", {})
                    t = next((v for k2, v in local_m.items() if k2.startswith("Trustworthiness")), None)
                    c = next((v for k2, v in local_m.items() if k2.startswith("Continuity")), None)
                    s = global_m.get("Stress")
                    if t is None or c is None or s is None:
                        continue
                    # composite: higher = easier to embed
                    rel_vals.append((float(t) + float(c) + (1.0 - float(s))) / 3.0)
                    scores[method] = {
                        "Trustworthiness": float(t),
                        "Continuity": float(c),
                        "Stress": float(s),
                    }
                except Exception as exc:
                    scores[method] = {"error": str(exc)}

            if rel_vals:
                patch_reliability_scores[pid] = float(np.mean(rel_vals))
            results.append(scores)

        # Map per-patch reliability back to per-point arrays
        # Fall back to the median of observed scores (not a hard-coded value)
        observed = list(patch_reliability_scores.values())
        fallback = float(np.median(observed)) if observed else 0.5
        point_reliability = np.array([
            patch_reliability_scores.get(int(pid), fallback)
            for pid in patch_labels
        ], dtype=float)

        # --- Relative / percentile-based difficulty classification ---
        # Use the 33rd and 67th percentiles of the *observed* distribution so
        # that labels are always relative to this dataset.  This prevents the
        # degenerate "all Hard" outcome when all patches happen to have low
        # absolute scores.
        q_low  = float(np.percentile(point_reliability, 33))
        q_high = float(np.percentile(point_reliability, 67))

        # If the score range is extremely narrow (< 0.02), keep the three
        # bands but centre them on the actual median so the UI still shows
        # meaningful colour variation.
        if q_high - q_low < 0.02:
            med = float(np.median(point_reliability))
            q_low  = med - 0.01
            q_high = med + 0.01

        point_difficulty = np.where(
            point_reliability >= q_high, "E",
            np.where(point_reliability <= q_low, "H", "M")
        )

        logger.info(
            "Patch difficulty thresholds — q33=%.3f  q67=%.3f  "
            "H=%d  M=%d  E=%d",
            q_low, q_high,
            (point_difficulty == "H").sum(),
            (point_difficulty == "M").sum(),
            (point_difficulty == "E").sum(),
        )
        return results, point_difficulty, point_reliability

    def _compute_neighborhood_grid(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Grid-search over neighbourhood sizes, embed, score."""
        from sklearn.cluster import MiniBatchKMeans

        n = data.shape[0]
        # Sample a few target points for efficiency
        sample_size = min(10, n // 50)
        rng = np.random.default_rng(42)
        target_indices = rng.choice(n, size=sample_size, replace=False)

        k_values = [20, 50, 100, 200, 500]
        results = []

        for idx in target_indices:
            dists = np.linalg.norm(data - data[idx], axis=1)
            for k in k_values:
                if k > n:
                    continue
                nn_idx = np.argsort(dists)[:k]
                patch = data[nn_idx]
                patch_lbl = labels[nn_idx] if labels is not None else None
                try:
                    emb = self.dr_engine.compute_embedding(patch, method="tsne", n_pcs=min(20, patch.shape[1]))
                    m = self.metrics_engine.compute_all(hd_data=patch, ld_data=emb, labels=patch_lbl)
                    results.append({
                        "target_point": int(idx),
                        "k": k,
                        "T": m.get("Trustworthiness"),
                        "C": m.get("Continuity"),
                        "S": m.get("Stress-1"),
                    })
                except Exception as exc:
                    results.append({"target_point": int(idx), "k": k, "error": str(exc)})

        return results

    def _compute_baseline(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Compute baseline embedding + metrics for outlier analysis."""
        method = self.config.default_dr_method
        hp = self._initial_hyperparams(method)
        emb = self.dr_engine.compute_embedding(data, method=method, **hp)
        m = self.metrics_engine.compute_all(hd_data=data, ld_data=emb, labels=labels,
                                             k=self._metrics_k,
                                             lof_neighbors=self._lof_neighbors)
        # Per-point LOF scores for the outlier overlay plot
        try:
            lof_scores = self.metrics_engine.compute_lof_scores(emb, n_neighbors=self._lof_neighbors)
        except Exception:
            lof_scores = None

        return {"metrics": m, "lof": m.get("LOF Statistics", {}),
                "embedding": emb, "lof_scores": lof_scores}

    # ------------------------------------------------------------------
    # Helpers — hyperparameters
    # ------------------------------------------------------------------

    def _initial_hyperparams(self, method: str) -> Dict[str, Any]:
        """Return full initial hyperparameter set from config."""
        cfg = self.config
        if method == "tsne":
            return {
                "perplexity": cfg.tsne_perplexity,
                "learning_rate": cfg.tsne_learning_rate,
                "max_iter": cfg.tsne_max_iter,
                "early_exaggeration": cfg.tsne_early_exaggeration,
                "angle": cfg.tsne_angle,
                "init": cfg.tsne_init,
                "metric": cfg.tsne_metric,
                "n_iter_without_progress": cfg.tsne_n_iter_without_progress,
                "n_pcs": cfg.n_pcs,
            }
        elif method == "umap":
            return {
                "n_neighbors": cfg.umap_n_neighbors,
                "min_dist": cfg.umap_min_dist,
                "metric": cfg.umap_metric,
                "n_epochs": cfg.umap_n_epochs,
                "learning_rate": cfg.umap_learning_rate,
                "spread": cfg.umap_spread,
                "negative_sample_rate": cfg.umap_negative_sample_rate,
                "repulsion_strength": cfg.umap_repulsion_strength,
                "set_op_mix_ratio": cfg.umap_set_op_mix_ratio,
                "local_connectivity": cfg.umap_local_connectivity,
                "n_pcs": cfg.n_pcs,
            }
        elif method == "pacmap":
            return {
                "n_neighbors": cfg.pacmap_n_neighbors,
                "MN_ratio": cfg.pacmap_MN_ratio,
                "FP_ratio": cfg.pacmap_FP_ratio,
                "lr": cfg.pacmap_lr,
                "num_iters": (cfg.pacmap_num_iters_phase1,
                              cfg.pacmap_num_iters_phase2,
                              cfg.pacmap_num_iters_phase3),
                "apply_pca": cfg.pacmap_apply_pca,
                "distance": cfg.pacmap_distance,
                "n_pcs": cfg.n_pcs,
            }
        else:
            return {"n_pcs": cfg.n_pcs}

    def _apply_recommendations(
        self, current: Dict[str, Any], opt_result: Dict[str, Any]
    ) -> tuple:
        """Parse Agent 6 recommendations and apply parameter updates.

        Returns (updated_params, applied_recs, skipped_recs).
        Now allows adding NEW parameters (for custom code) instead of
        silently skipping them.
        """
        updated = copy.deepcopy(current) if isinstance(current, dict) else {}
        if not isinstance(opt_result, dict):
            logger.warning("Optimizer output was %s instead of dict; skipping recommendations", type(opt_result).__name__)
            return updated, [], []
        recs = opt_result.get("recommendations", [])
        if isinstance(recs, dict):
            recs = [recs]
        elif not isinstance(recs, list):
            recs = []
        applied = []
        skipped = []

        for rec in recs:
            param = rec.get("parameter", "")
            suggested = rec.get("suggested_value")
            if not param or suggested is None:
                continue
            try:
                if param in updated:
                    # Existing param — type-safe conversion
                    if isinstance(updated[param], int):
                        updated[param] = int(float(suggested))
                    elif isinstance(updated[param], float):
                        updated[param] = float(suggested)
                    else:
                        updated[param] = suggested
                else:
                    # New param — infer type; custom code may reference it
                    if isinstance(suggested, str):
                        # Try numeric conversion
                        try:
                            val = int(suggested)
                        except (ValueError, TypeError):
                            try:
                                val = float(suggested)
                            except (ValueError, TypeError):
                                val = suggested
                        updated[param] = val
                    else:
                        updated[param] = suggested
                    logger.info("  NEW param %s = %s", param, updated[param])

                applied.append(rec)
                logger.info("  %s: %s → %s", param, rec.get("current_value"), suggested)
            except (ValueError, TypeError):
                skipped.append(rec)
                logger.warning("  Could not apply recommendation for %s: %s", param, suggested)

        return updated, applied, skipped

    # ------------------------------------------------------------------
    # Helpers — labels
    # ------------------------------------------------------------------

    @staticmethod
    def _label_summary(
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]] = None,
    ) -> Optional[str]:
        if labels is None:
            return None
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        parts = []
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            name = label_names.get(int(u), str(u)) if label_names else str(u)
            parts.append(f"{name}: {c} ({100*c/total:.1f}%)")
        return "; ".join(parts)
