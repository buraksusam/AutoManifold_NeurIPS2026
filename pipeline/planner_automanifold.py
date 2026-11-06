"""
AutoManifoldPlanner — orchestrates the AutoManifold algorithm-design pipeline.

Generates genuinely novel dimensionality reduction algorithms through
template-based composition of algorithmic primitives (affinity, cost,
optimization). The analysis agents follow DR_Template (docs/DR_Template.md);
the final synthesis (Agent 8) follows DR_Template_v2 (docs/DR_Template_v2.md).

Phase 0: Primitive Selection
Phase 1: Data Characterization (Agent 1)
Phase 2: Difficulty-Aware Primitive Matching (Agent 2, Agent 3)
Phase 3: Algorithm Composition (Agent 4, Agent 5, Agent 6)
Phase 4: Implementation & Validation (Agent 7, Code Generation)
Phase 5: Iterative Refinement (if needed)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re as _re
import signal
import time
import traceback
from contextlib import contextmanager
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
    plot_distortion_heatmap,
    plot_highlighted_areas,
    plot_class_distortion_summary,
    plot_convergence_curve,
    plot_dendrogram_comparison,
    plot_hierarchy_change_detail,
)

# Novel algorithm agents
from .agents.agent_1_data_characterization import DataCharacterizationAgent
from .agents.agent_2_difficulty_analyzer import EmbeddingDifficultyAgent
from .agents.agent_3_neighborhood_analyzer import NeighborhoodAnalyzerAgent
from .agents.agent_4_affinity_designer import AffinityDesignerAgent
from .agents.agent_5_cost_composer import CostComposerAgent
from .agents.agent_6_hyperparameter_opt import HyperparameterOptAgent
from .agents.agent_12_novelty_validator import NoveltyValidatorAgent
from .agents.agent_7_visualization_evaluator import VisualizationEvaluatorAgent
from .agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
from .agents.agent_9_visual_forensics import VisualForensicsAgent
from .agents.agent_10_error_recovery import ErrorRecoveryAgent
from .agents.agent_11_math_pseudocode import MathPseudocodeAgent

logger = logging.getLogger(__name__)

# Wall-clock ceiling for a single custom_embedding() execution. A pathological
# LLM-generated algorithm (e.g. a near-dense SNN graph optimized with per-edge
# np.add.at scatter) can otherwise run for hours per iteration. Overridable via
# the DR_EMBEDDING_TIMEOUT env var (seconds); 0/negative disables the guard.
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = int(os.environ.get("DR_EMBEDDING_TIMEOUT", "600"))


class EmbeddingTimeout(Exception):
    """Raised when a custom_embedding call exceeds the wall-clock limit."""


@contextmanager
def _time_limit(seconds):
    """Abort the wrapped block after `seconds` via SIGALRM. SIGALRM fires at the
    next Python bytecode boundary, so a long numpy loop is interrupted between
    iterations. No-op when disabled or when not on the main thread (SIGALRM is
    main-thread-only, e.g. under a threaded web server)."""
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise EmbeddingTimeout(f"custom_embedding exceeded {seconds}s wall-clock limit")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Not on the main thread — cannot install a signal handler here.
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class AutoManifoldPlanner:
    """
    AutoManifold: novel DR algorithm generation by template-based composition
    of algorithmic primitives. Also the base class for NoSkel-Agents and
    SingleLLM, which reuse its execution, evaluation and forensics helpers.
    """

    def __init__(self, config: PipelineConfig, disabled_components=None):
        self.config = config
        # Component-ablation hook: names in this set skip their phase /
        # prompt injection.
        self.disabled_components = set(disabled_components or [])
        # "hyperparam_opt" ablation: disable the iterative optimization loop so
        # only the initial designed algorithm is evaluated (no refinement).
        if "hyperparam_opt" in self.disabled_components:
            self.config.convergence.max_iterations = 1
        # Wall-clock ceiling per custom_embedding execution (guards against
        # pathological generated algorithms hanging a run for hours).
        self._embedding_timeout = DEFAULT_EMBEDDING_TIMEOUT_SECONDS
        self.llm = LLMClient(config.llm)
        self.mode_name = "automanifold"
        self.template_name = "DR_Template_v2"

        # Compute engines
        self.dr_engine = DRVisualizationEngine()
        self.metrics_engine = QuantitativeAnalystEngine()
        self._metrics_k = config.metrics.neighborhood_k
        self._lof_neighbors = config.metrics.lof_neighbors
        self.clustering_engine = ClusteringEngine()

        # Worker agents
        self.agent1 = DataCharacterizationAgent(self.llm)
        self.agent2 = EmbeddingDifficultyAgent(self.llm)
        self.agent3 = NeighborhoodAnalyzerAgent(self.llm)
        self.agent4 = AffinityDesignerAgent(self.llm)
        self.agent5 = CostComposerAgent(self.llm)
        self.agent6 = HyperparameterOptAgent(self.llm)        # iterative optimizer
        self.agent12_novelty = NoveltyValidatorAgent(self.llm) # Phase 4 novelty
        self.agent7_eval = VisualizationEvaluatorAgent(self.llm)  # Phase 6 evaluator
        self.agent8 = AlgorithmSynthesizerAgent(self.llm, mode="automanifold")
        self.agent9 = VisualForensicsAgent(self.llm)     # visual forensics
        self.agent10 = ErrorRecoveryAgent(self.llm)        # code error recovery
        self.agent_math = MathPseudocodeAgent(self.llm)    # math representation + pseudocode

        # Pipeline state
        self.context: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.output_dir: str = OUTPUT_ROOT

    @staticmethod
    def _stage(icon: str, phase: str, detail: str = "") -> None:
        """Print a clean one-line stage banner."""
        suffix = f"  › {detail}" if detail else ""
        print(f"\n{'─'*60}", flush=True)
        print(f"  {icon}  {phase}{suffix}", flush=True)
        print(f"{'─'*60}", flush=True)

    def _save_json(self, filename: str, data: Any) -> str:
        """Save dict to JSON file in output directory."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def _phase_math_pseudocode(
        self,
        algorithm_name: str,
        code_str: str,
        default_params: Optional[Dict[str, Any]] = None,
        description: str = "",
        output_prefix: str = "novel",
    ) -> Dict[str, Any]:
        """Generate mathematical representation + pseudocode for the final algorithm.

        Saves <output_prefix>_math_pseudocode.json and .md alongside the run
        outputs. Always returns the parsed dict (possibly empty on failure).
        Never raises — math/pseudocode is auxiliary documentation, not gating.
        """
        if not code_str:
            print("  ⚠️  No code available — skipping math/pseudocode layer.", flush=True)
            empty = {"skipped": True, "reason": "no code available"}
            self._save_json(f"{output_prefix}_math_pseudocode.json", empty)
            return empty

        print(f"  [1/1] Generating math + pseudocode for {algorithm_name!r} …", flush=True)
        try:
            spec = self.agent_math.run({
                "algorithm_name": algorithm_name,
                "description": description,
                "code": code_str,
                "default_params": default_params or {},
            })
        except Exception as exc:
            logger.warning("Math/pseudocode agent failed: %s", exc)
            spec = {"error": str(exc)}

        if not spec or spec.get("error"):
            print("  ⚠️  Math/pseudocode generation failed; saving stub.", flush=True)
            self._save_json(f"{output_prefix}_math_pseudocode.json",
                            {"algorithm_name": algorithm_name, **(spec or {})})
            return spec or {}

        spec.setdefault("algorithm_name", algorithm_name)
        self._save_json(f"{output_prefix}_math_pseudocode.json", spec)
        try:
            md = MathPseudocodeAgent.render_markdown(spec)
            md_path = os.path.join(self.output_dir, f"{output_prefix}_math_pseudocode.md")
            with open(md_path, "w") as f:
                f.write(md)
            print(f"  → Math/pseudocode saved: {md_path}", flush=True)
        except Exception as exc:
            logger.warning("Markdown render failed: %s", exc)
        return spec

    def run(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_names: Optional[Dict[int, str]] = None,
        output_dir: str = OUTPUT_ROOT,
    ) -> Dict[str, Any]:
        """Execute the novel algorithm generation pipeline on *data* (n × D)."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        n, d = data.shape

        # Available primitives for novel algorithm generation
        available_primitives = {
            "affinity_primitives": [
                "euclidean_knn",
                "mutual_knn",
                "snn",  # shared nearest neighbor
                "geodesic",
                "lof_weighted",
                "learnable",
                "perplexity_gaussian",  # t-SNE style per-point calibrated Gaussian
                "umap_fuzzy"            # UMAP fuzzy-simplicial local membership
            ],
            "cost_primitives": [
                "kl_divergence",
                "cross_entropy",
                "spectral",
                "force_directed",
                "mutual_information"
            ],
            "ld_kernel_primitives": [
                "student_t",     # (1 + ||y_i - y_j||^2 / dof)^(-(dof+1)/2); dof=1 = t-SNE
                "gaussian_ld",   # exp(-||y_i - y_j||^2); SNE-style, suffers crowding
                "umap_ab",       # 1 / (1 + a * ||y_i - y_j||^(2b)); UMAP fit-to-min-dist
            ],
            "optimization_primitives": [
                "sgd",
                "adam",
                "spectral_eigen",
                "fft_accelerated"
            ]
        }

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
                "novelty_target": "compositional_novelty",
            },
            "available_methods": {
                "clustering_algorithms": ["kmeans", "agglomerative", "spectral"],
            },
            "available_primitives": (
                {} if "primitive_library" in self.disabled_components
                else available_primitives
            ),
            # Exposed so design agents can strip their primitive-library /
            # composition-pattern prompt blocks for the ablation.
            "disabled_components": sorted(self.disabled_components),
        }
        if "primitive_library" in self.disabled_components:
            logger.info("[ablation] primitive_library disabled — empty primitive vocabulary")

        self._stage("🧪", "PHASE 0 — Baseline Calibration")
        self._phase0_baseline_calibration(data, labels)

        self._stage("📊", "PHASE 1 — Data Characterization")
        self._phase1_characterize(data, labels)

        self._stage("🔍", "PHASE 2 — Difficulty-Aware Primitive Matching")
        self._phase2_difficulty_analysis(data, labels)

        self._stage("🎨", "PHASE 3 — Algorithm Composition")
        self._phase3_compose_algorithm()

        self._stage("✨", "PHASE 4 — Novelty Validation")
        self._phase4_validate_novelty()

        self._stage("🏗️", "PHASE 5 — Algorithm Synthesis & Implementation")
        final = self._phase5_synthesize()

        self._stage("🚀", "PHASE 6 — Implementation & Evaluation")
        eval_result = self._phase6_implement_and_evaluate(data, labels, label_names, final)
        final["implementation_result"] = eval_result

        self._stage("📐", "PHASE 6b — Mathematical Representation & Pseudocode")
        if eval_result.get("used_fallback"):
            # Phase 6 fell back to t-SNE because the synthesized code was
            # unrecoverable. There is no novel algorithm to describe — running
            # MathPseudocodeAgent over an empty code blob produces a misleading
            # report. Skip and record the reason.
            print("  ⏭  Skipped — Phase 6 fell back to t-SNE; no novel code to describe.", flush=True)
            final["math_pseudocode"] = {"skipped": "fallback_to_tsne"}
        else:
            novel_spec = final.get("novel_algorithm_spec", {}) if isinstance(final, dict) else {}
            # Prefer the post-recovery executed code stashed by Phase 6; fall back
            # to the original Agent-8 synthesis if Phase 6 did not store it.
            novel_code = eval_result.get("code") or novel_spec.get("custom_embedding_code", "")
            novel_params = eval_result.get("params") or novel_spec.get("default_params", {})
            novel_desc = ""
            if isinstance(novel_spec, dict):
                algo_spec = novel_spec.get("algorithm_specification", {}) or {}
                novel_desc = algo_spec.get("description") or algo_spec.get("brief_description") or ""
            final["math_pseudocode"] = self._phase_math_pseudocode(
                algorithm_name=getattr(self, "_novel_algo_name", "Novel Algorithm"),
                code_str=novel_code,
                default_params=novel_params,
                description=novel_desc,
                output_prefix="novel",
            )

        self._stage("🔬", "PHASE 7 — Visual Forensics")
        if eval_result.get("used_fallback"):
            # Forensics over a t-SNE fallback would be mislabelled as the
            # novel algorithm's behaviour. Skip and surface the reason in
            # the final report so consumers don't read stale plots.
            print("  ⏭  Skipped — fallback embedding is t-SNE, not the novel algorithm.", flush=True)
            vi_result = {"skipped": "fallback_to_tsne"}
        else:
            vi_result = self._phase7_visual_forensics(data, labels, label_names)
        final["visual_forensics"] = vi_result
        final["mode"] = self.mode_name
        final["template_version"] = self.template_name

        # Persist final report
        report_path = os.path.join(output_dir, "final_novel_algorithm_report.json")
        with open(report_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final report saved → %s", report_path)

        return final

    # ------------------------------------------------------------------
    # Phase 0 — Baseline Calibration
    # ------------------------------------------------------------------
    # Runs t-SNE / UMAP / PACMAP on the full dataset, scores them by the
    # user's primary_goal, and exposes a `baseline_profile` that downstream
    # agents use as a SOFT BIAS. Design agents may override, but the winning
    # baseline's recipe becomes the default anchor.

    # Method → canonical recipe anchor. Deviations by design agents are allowed.
    _METHOD_RECIPE_ANCHOR = {
        "tsne": {
            "family": "tsne",
            "hd_affinity": "perplexity_gaussian",
            "ld_kernel": {"primitive": "student_t", "parameters": {"dof": 1}},
            "cost": "kl_divergence",
        },
        "umap": {
            "family": "umap",
            "hd_affinity": "umap_fuzzy",
            "ld_kernel": {"primitive": "umap_ab",
                          "parameters": {"a": 1.577, "b": 0.895}},
            "cost": "cross_entropy",
        },
        "pacmap": {
            "family": "pacmap",
            "hd_affinity": "mutual_knn",
            "ld_kernel": {"primitive": "umap_ab",
                          "parameters": {"a": 1.0, "b": 1.0}},
            "cost": "mixed_triplet",
        },
    }

    @staticmethod
    def _score_baseline(metrics: Dict[str, Any], intent: str) -> float:
        """Weighted score of a baseline's metrics under the user's primary_goal."""
        T = float(metrics.get("Trustworthiness", 0.0) or 0.0)
        C = float(metrics.get("Continuity", 0.0) or 0.0)
        S = float(metrics.get("Stress", 1.0) or 1.0)
        Sp = float(metrics.get("Spearman", 0.0) or 0.0)
        Sil = float(metrics.get("Silhouette", 0.0) or 0.0)
        LOFpct = float(metrics.get("LOF_Outlier_pct", 100.0) or 100.0)
        # AUC variants (k-sweep). Fall back to single-k T/C when the sweep is
        # missing (small datasets, or runs predating the AUC engine update).
        T_auc = float(metrics.get("Trustworthiness_AUC", 0.0) or 0.0) or T
        C_auc = float(metrics.get("Continuity_AUC", 0.0) or 0.0) or C
        s_norm = 1.0 - min(max(S, 0.0), 1.0)           # stress → [0,1], higher=better
        sil_norm = 0.5 * (max(-1.0, min(1.0, Sil)) + 1.0)   # [-1,1] → [0,1]
        lof_norm = 1.0 - min(max(LOFpct, 0.0), 100.0) / 100.0

        if intent == "local":
            return 0.7 * T + 0.3 * C                   # user-confirmed weights
        if intent == "local_auc":
            return 0.7 * T_auc + 0.3 * C_auc
        if intent == "global":
            return 0.5 * Sp + 0.5 * s_norm
        if intent == "trustworthiness":
            return T
        if intent == "trustworthiness_auc":
            return T_auc
        if intent == "continuity":
            return C
        if intent == "continuity_auc":
            return C_auc
        if intent == "spearman":
            return max(0.0, min(1.0, Sp))
        if intent == "silhouette":
            return sil_norm
        if intent == "stress":
            return s_norm
        if intent == "lof":
            return lof_norm
        if intent == "balanced_auc":
            return 0.25 * (T_auc + C_auc + Sp + s_norm)
        # balanced
        return 0.25 * (T + C + Sp + s_norm)

    @staticmethod
    def _infer_intent(primary_goal: str) -> str:
        g = (primary_goal or "").lower().strip()
        # AUC-flavoured intents take precedence over their single-k cousins so
        # phrases like "maximum trustworthiness AUC" / "local AUC" route to the
        # k-sweep scorer instead of the single-k one.
        auc_requested = "auc" in g or "k-sweep" in g or "k sweep" in g

        # Group / balanced goals are detected BEFORE the single-metric
        # substring checks because their descriptive prompts may name several
        # metrics (e.g. "balance local and global structure — ...
        # Trustworthiness@k, Continuity@k, Spearman ...") without making any
        # single one the SOLE optimization target.
        if g.startswith("balance"):
            return "balanced_auc" if auc_requested else "balanced"
        if g.startswith("local structure") or g.startswith("local "):
            return "local_auc" if auc_requested else "local"
        if g.startswith("global structure") or g.startswith("global "):
            return "global"
        if g == "local":
            return "local_auc" if auc_requested else "local"
        if g == "global":
            return "global"

        # Explicit single-metric goals (typically "maximum X" / "minimum
        # stress") fall through to substring matching.
        if "trustworthiness" in g:
            return "trustworthiness_auc" if auc_requested else "trustworthiness"
        if "continuity" in g:
            return "continuity_auc" if auc_requested else "continuity"
        if "spearman" in g:
            return "spearman"
        if "silhouette" in g:
            return "silhouette"
        if "lof" in g:
            return "lof"
        if "stress" in g:
            return "stress"
        if "local" in g:
            return "local_auc" if auc_requested else "local"
        if auc_requested:
            return "balanced_auc"
        if "global" in g or "topology" in g or "skeleton" in g:
            return "global"
        return "balanced"

    def _phase0_baseline_calibration(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> None:
        """Run baselines, score them, and populate context['baseline_profile']."""
        print("  [1/1] Running baselines (tsne, umap, pacmap) …", flush=True)

        intent = self._infer_intent(self.config.primary_goal)
        methods = ["tsne", "umap", "pacmap"]
        metrics_per_method: Dict[str, Dict[str, float]] = {}
        embeddings_path: Dict[str, str] = {}

        for method in methods:
            try:
                emb = self.dr_engine.compute_embedding(data, method=method)
                m_nested = self.metrics_engine.compute_all(
                    hd_data=data, ld_data=emb, labels=labels,
                    k=self._metrics_k, lof_neighbors=self._lof_neighbors,
                )
                # compute_all returns a NESTED dict; flatten via the existing helper
                snap = self._extract_metrics_snapshot(m_nested)
                def _as_float(v, default):
                    try:
                        if v is None:
                            return default
                        return float(v)
                    except (TypeError, ValueError):
                        return default
                metrics_per_method[method] = {
                    "Trustworthiness": _as_float(snap.get("Trustworthiness"), 0.0),
                    "Continuity":      _as_float(snap.get("Continuity"), 0.0),
                    "Trustworthiness_AUC": _as_float(snap.get("Trustworthiness_AUC"), 0.0),
                    "Continuity_AUC":      _as_float(snap.get("Continuity_AUC"), 0.0),
                    "k_sweep":         snap.get("k_sweep"),
                    "Trustworthiness_curve": snap.get("Trustworthiness_curve"),
                    "Continuity_curve":      snap.get("Continuity_curve"),
                    "Stress":          _as_float(snap.get("Stress"), 1.0),
                    "Spearman":        _as_float(snap.get("Spearman"), 0.0),
                    "Silhouette":      _as_float(snap.get("Silhouette"), 0.0),
                }
                emb_path = os.path.join(self.output_dir, f"baseline_{method}.npy")
                np.save(emb_path, emb)
                embeddings_path[method] = emb_path
            except Exception as exc:
                print(f"  ⚠️  Baseline {method} failed: {exc}", flush=True)
                metrics_per_method[method] = {"error": str(exc)}

        self._build_and_save_baseline_profile(
            metrics_per_method, embeddings_path, methods, intent,
        )

    def _build_and_save_baseline_profile(
        self,
        metrics_per_method: Dict[str, Dict[str, float]],
        embeddings_path: Dict[str, str],
        methods: List[str],
        intent: str,
    ) -> None:
        """Score precomputed baseline metrics under *intent* and persist the
        ``baseline_profile`` to context + disk.

        Extracted from ``_phase0_baseline_calibration`` so cache-aware planners
        can feed in metrics that were computed in an earlier run for the same
        dataset and skip the heavy DR+metrics step.
        """
        # Score and pick winner
        scored = {
            m: self._score_baseline(v, intent)
            for m, v in metrics_per_method.items() if "error" not in v
        }
        if not scored:
            print("  ⚠️  All baselines failed; skipping calibration.", flush=True)
            return
        winner_name = max(scored, key=scored.get)
        winner_metrics = metrics_per_method[winner_name]

        # Scoring formula description (for prompts)
        scoring_formula = {
            "local":              "0.7·T + 0.3·C",
            "local_auc":          "0.7·T_AUC + 0.3·C_AUC  (k-sweep AUC)",
            "global":             "0.5·Spearman + 0.5·(1 - min(Stress, 1))",
            "balanced":           "0.25 each on T, C, Spearman, 1-min(Stress,1)",
            "balanced_auc":       "0.25 each on T_AUC, C_AUC, Spearman, 1-min(Stress,1)",
            "trustworthiness":    "T (Trustworthiness@k only)",
            "trustworthiness_auc":"T_AUC (Trustworthiness AUC over k-sweep)",
            "continuity":         "C (Continuity@k only)",
            "continuity_auc":     "C_AUC (Continuity AUC over k-sweep)",
            "spearman":           "Spearman (HD↔LD distance rank correlation)",
            "silhouette":         "(Silhouette + 1)/2",
            "stress":             "1 - min(Stress, 1)  (lower stress = higher score)",
            "lof":                "1 - LOF_Outlier_pct/100  (fewer LOF outliers = higher score)",
        }.get(intent, "0.25 each on T, C, Spearman, 1-min(Stress,1)")

        # Weakness analysis: which metric is the winner worst at, and where does
        # a runner-up beat it? For explicit-metric intents, the "weakness" IS the
        # target metric — the design must push it higher regardless of where the
        # baseline already sits.
        explicit_intent_to_metric = {
            "trustworthiness":     "Trustworthiness",
            "trustworthiness_auc": "Trustworthiness_AUC",
            "continuity":          "Continuity",
            "continuity_auc":      "Continuity_AUC",
            "spearman":            "Spearman",
            "silhouette":          "Silhouette",
            "stress":              "Stress",
            "lof":                 "LOF_Outlier_pct",
        }
        if intent in explicit_intent_to_metric:
            weakness_metric = explicit_intent_to_metric[intent]
        elif intent in {"local_auc", "balanced_auc"}:
            weakness_metric = min(
                ["Trustworthiness_AUC", "Continuity_AUC", "Spearman"],
                key=lambda k: winner_metrics.get(k, 1.0),
            )
        else:
            weakness_metric = min(
                ["Trustworthiness", "Continuity", "Spearman"],
                key=lambda k: winner_metrics.get(k, 1.0),
            )
        runner_ups_wins: List[str] = []
        runner_up_scan_metrics = (
            ["Trustworthiness_AUC", "Continuity_AUC", "Spearman", "Silhouette"]
            if intent in {"local_auc", "balanced_auc",
                          "trustworthiness_auc", "continuity_auc"}
            else ["Trustworthiness", "Continuity", "Spearman", "Silhouette"]
        )
        for m in runner_up_scan_metrics:
            for other, om in metrics_per_method.items():
                if other == winner_name or "error" in om:
                    continue
                if om.get(m, -np.inf) > winner_metrics.get(m, -np.inf) + 1e-4:
                    runner_ups_wins.append(f"{other} > {winner_name} on {m}")

        beat_targets = {
            "Trustworthiness": winner_metrics["Trustworthiness"],
            "Continuity":      winner_metrics["Continuity"],
            "Trustworthiness_AUC": winner_metrics.get("Trustworthiness_AUC", 0.0),
            "Continuity_AUC":      winner_metrics.get("Continuity_AUC", 0.0),
            "Stress":          winner_metrics["Stress"],
            "Spearman":        winner_metrics["Spearman"],
            "tolerance":       0.005,
        }

        recipe_anchor = self._METHOD_RECIPE_ANCHOR.get(winner_name, {})

        profile = {
            "primary_goal":   self.config.primary_goal,
            "intent":         intent,
            "methods_evaluated": methods,
            "metrics_per_method": metrics_per_method,
            "embeddings_path": embeddings_path,
            "winner": {
                "name": winner_name,
                "primary_goal_score": round(scored[winner_name], 4),
                "all_scores": {m: round(s, 4) for m, s in scored.items()},
                "scoring_formula": f"{intent}: {scoring_formula}",
            },
            "recipe_anchor": recipe_anchor,
            "beat_targets":  beat_targets,
            "weakness_analysis": {
                "winner_weakest_metric": weakness_metric,
                "runner_up_wins":        runner_ups_wins,
                "augmentation_focus": (
                    f"Winner is {winner_name}. Its weakest axis is {weakness_metric}. "
                    f"Novel should anchor on the {recipe_anchor.get('family','?')} "
                    f"recipe and add a lightweight term targeting {weakness_metric} "
                    "without disturbing the primary objective."
                ),
            },
        }

        self.context["baseline_profile"] = profile
        p = self._save_json("baseline_profile.json", profile)
        print(f"  → Winner: {winner_name}  (score={scored[winner_name]:.4f} under intent='{intent}')", flush=True)
        print(f"  → Beat targets: T={beat_targets['Trustworthiness']:.4f}  "
              f"C={beat_targets['Continuity']:.4f}  "
              f"S={beat_targets['Stress']:.4f}  "
              f"Sp={beat_targets['Spearman']:.4f}", flush=True)
        print(f"  → Saved: {p}", flush=True)

    def _phase1_characterize(self, data: np.ndarray, labels: Optional[np.ndarray]):
        """Run Agent 1 — Data Characterization with primitive recommendations."""
        from sklearn.decomposition import PCA

        print("  [1/1] Characterizing data and recommending primitives …", flush=True)

        # Compute PCA for intrinsic dimensionality
        pca_full = PCA().fit(data)
        explained_10 = float(np.sum(pca_full.explained_variance_ratio_[:10]))
        explained_20 = float(np.sum(pca_full.explained_variance_ratio_[:20]))
        explained_50 = float(np.sum(pca_full.explained_variance_ratio_[:min(50, data.shape[1])]))

        self.context["data_summary"] = json.dumps({
            "pca_explained_variance": {
                "10": round(explained_10, 4),
                "20": round(explained_20, 4),
                "50": round(explained_50, 4)
            },
        })
        result = self.agent1.run(self.context)
        self.context["agent1_output"] = result
        p = self._save_json("agent1_data_characterization.json", result)
        print(f"  → Saved: {p}", flush=True)

        # Display primitive recommendations
        prim_recs = result.get("recommendations", {}).get("primitive_recommendations", {})
        affinity_candidates = prim_recs.get("affinity_candidates", [])
        cost_candidates = prim_recs.get("cost_candidates", [])

        if affinity_candidates:
            print(f"  → Recommended affinities: {', '.join([a.get('primitive', '?') for a in affinity_candidates[:3]])}", flush=True)
        if cost_candidates:
            print(f"  → Recommended costs: {', '.join([c.get('primitive', '?') for c in cost_candidates[:3]])}", flush=True)

    def _phase2_difficulty_analysis(self, data: np.ndarray, labels: Optional[np.ndarray]):
        """Run Agents 2 and 3 — Difficulty classification with primitive mapping."""
        # Agent 2: Difficulty classification + primitive mapping
        print("  [1/2] Computing difficulty and mapping primitives (Agent 2) …", flush=True)

        # Compute patch scores for difficulty classification.
        # Only `patch_scores` is consumed downstream (Agent 2). The per-point
        # H/M/E and reliability arrays are kept as locals — Optimize-mode and
        # the Skills planner produce/consume their own copies.
        patch_scores, _point_difficulty, _point_reliability = self._compute_patch_scores(data, labels)
        self.context["patch_scores"] = json.dumps(patch_scores, indent=2, default=str)

        if "difficulty_classifier" in self.disabled_components:
            logger.info("[ablation] difficulty_classifier disabled — skipping Agent 2")
            result2 = {}
        else:
            result2 = self.agent2.run(self.context)
        self.context["agent2_output"] = result2
        p = self._save_json("agent2_difficulty_primitive_mapping.json", result2)
        print(f"  → Saved: {p}", flush=True)

        # Display primitive mapping. Agent 2 emits `auxiliary_affinity`; older
        # prompt versions used `recommended_affinity` — accept either.
        prim_mapping = result2.get("difficulty_primitive_mapping", {})
        for region in ["H_regions", "M_regions", "E_regions"]:
            if region in prim_mapping:
                entry = prim_mapping[region] or {}
                affinity = (
                    entry.get("auxiliary_affinity")
                    or entry.get("recommended_affinity")
                    or "?"
                )
                weight = entry.get("auxiliary_blend_weight")
                suffix = f"  (aux weight={weight})" if weight is not None else ""
                print(f"  → {region}: {affinity}{suffix}", flush=True)

        # Agent 3: Maximal neighborhoods (can run in parallel)
        print("  [2/2] Estimating maximal neighborhoods (Agent 3) …", flush=True)
        grid_results = self._compute_neighborhood_grid(data, labels)
        self.context["grid_search_results"] = json.dumps(grid_results, indent=2, default=str)

        if "neighborhood_analyzer" in self.disabled_components:
            logger.info("[ablation] neighborhood_analyzer disabled — skipping Agent 3")
            result3 = {}
        else:
            result3 = self.agent3.run(self.context)
        self.context["agent3_output"] = result3
        p = self._save_json("agent3_maximal_neighborhoods.json", result3)
        print(f"  → Saved: {p}", flush=True)

        opt_k = result3.get("maximal_neighborhoods", {}).get("global_estimate", {}).get("optimal_k", "?")
        print(f"  → Optimal max neighborhood k ≈ {opt_k}", flush=True)

    def _phase3_compose_algorithm(self):
        """Run Agents 4 and 5 — Compose affinity and cost designs."""
        # Agent 4: Affinity Graph Designer
        print("  [1/2] Designing affinity graph (Agent 4) …", flush=True)
        if "affinity_designer" in self.disabled_components:
            logger.info("[ablation] affinity_designer disabled — skipping Agent 4")
            result4 = {}
        else:
            result4 = self.agent4.run(self.context)
        self.context["agent4_output"] = result4
        p = self._save_json("agent4_affinity_design.json", result4)
        print(f"  → Saved: {p}", flush=True)

        affinity_design = result4.get("affinity_design", {})
        primary = affinity_design.get("primary_primitive", "?")
        is_hybrid = affinity_design.get("is_hybrid", False)
        print(f"  → Primary affinity: {primary} (hybrid={is_hybrid})", flush=True)

        # Agent 5: Cost Function Composer
        print("  [2/2] Composing cost function (Agent 5) …", flush=True)
        if "cost_composer" in self.disabled_components:
            logger.info("[ablation] cost_composer disabled — skipping Agent 5")
            result5 = {}
        else:
            result5 = self.agent5.run(self.context)
        self.context["agent5_output"] = result5
        p = self._save_json("agent5_cost_composition.json", result5)
        print(f"  → Saved: {p}", flush=True)

        cost_design = result5.get("cost_design", {})
        objective = cost_design.get("primary_objective", "?")
        is_multi = cost_design.get("is_multiobjective", False)
        print(f"  → Primary objective: {objective} (multi-objective={is_multi})", flush=True)

        # Create a simple optimization design from the components
        self.context["agent6_output"] = self._design_optimization(result4, result5)

    def _design_optimization(self, affinity_result: dict, cost_result: dict) -> dict:
        """Design optimization strategy based on affinity and cost designs."""
        affinity_design = affinity_result.get("affinity_design", {})
        cost_design = cost_result.get("cost_design", {})

        computational_cost = affinity_result.get("expected_properties", {}).get("computational_cost", "medium")
        is_multiobjective = cost_design.get("is_multiobjective", False)

        # Select optimization method based on cost type and computational budget
        primary_objective = cost_design.get("primary_objective", "")

        if "spectral" in primary_objective.lower():
            method = "spectral_eigen"
        elif computational_cost == "high":
            method = "fft_accelerated"
        else:
            method = "adam"

        optimization_design = {
            "optimization_design": {
                "method": method,
                "hyperparameters": {
                    "learning_rate": 0.1 if method == "sgd" else 0.01,
                    "max_iterations": 1000,
                    "convergence_tolerance": 1e-5,
                },
                "convergence_criteria": ["score_improvement < 0.01", "max_iterations"],
                "approximation_methods": ["nn_descent"] if computational_cost == "high" else []
            },
            "expected_performance": {
                "time_complexity": "O(n log n)" if method == "spectral_eigen" else "O(n^2)",
                "memory_requirements": "medium" if computational_cost != "high" else "high",
                "scalability": "good" if method in ["adam", "sgd"] else "limited"
            }
        }

        return optimization_design

    def _phase4_validate_novelty(self):
        """Run Agent 12 — Validate algorithm novelty."""
        print("  [1/1] Assessing algorithm novelty (Agent 12) …", flush=True)

        if "novelty_validator" in self.disabled_components:
            logger.info("[ablation] novelty_validator disabled — skipping Agent 12")
            result12 = {}
        else:
            result12 = self.agent12_novelty.run(self.context)
        self.context["agent12_output"] = result12
        p = self._save_json("agent12_novelty_validation.json", result12)
        print(f"  → Saved: {p}", flush=True)

        most_similar = result12.get("most_similar_method", "?")
        theoretical_diff = result12.get("theoretical_differences", "")
        is_novel = result12.get("is_genuinely_novel", False)
        suggested_name = result12.get("suggested_name", "")

        print(f"  → Most similar method: {most_similar}  (genuinely novel: {is_novel})", flush=True)
        if theoretical_diff:
            # Trim to one line for console; full text is in the saved JSON.
            summary = theoretical_diff.replace("\n", " ").strip()
            if len(summary) > 220:
                summary = summary[:217] + "…"
            print(f"  → Theoretical differences: {summary}", flush=True)
        if suggested_name:
            print(f"  → Suggested name: {suggested_name}", flush=True)

    def _phase5_synthesize(self) -> Dict[str, Any]:
        """Run Agent 8 — Synthesize final algorithm specification."""
        print("  [1/1] Synthesizing final algorithm (Agent 8) …", flush=True)

        final_spec = self.agent8.run(self.context)
        p = self._save_json("agent8_final_synthesis.json", final_spec)
        print(f"  → Saved: {p}", flush=True)

        name = final_spec.get("algorithm_specification", {}).get("name", "—")
        print(f"  → Final algorithm: {name}", flush=True)

        return {
            "novel_algorithm_spec": final_spec,
            "agent_outputs": {
                "data_characterization": self.context.get("agent1_output"),
                "difficulty_mapping": self.context.get("agent2_output"),
                "neighborhoods": self.context.get("agent3_output"),
                "affinity_design": self.context.get("agent4_output"),
                "cost_design": self.context.get("agent5_output"),
                "optimization_design": self.context.get("agent6_output"),
                "novelty_validation": self.context.get("agent12_output"),
            },
            "novelty_report": self.context.get("agent12_output", {}).get("novelty_assessment", {}),
        }

    # ------------------------------------------------------------------
    # Phase 6 — Implementation, Iterative Optimisation & Evaluation
    # ------------------------------------------------------------------

    def _phase6_implement_and_evaluate(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        phase5_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute, iteratively optimise, and evaluate the synthesized algorithm."""
        spec = phase5_result.get("novel_algorithm_spec", {})
        code_str = spec.get("custom_embedding_code", "")
        default_params = spec.get("default_params", {})
        algo_name = spec.get("algorithm_specification", {}).get("name", "Novel Algorithm")
        self._novel_algo_name = algo_name   # store for Phase 7

        result: Dict[str, Any] = {"algorithm_name": algo_name, "success": False}

        # --- 1. Extract and validate code ---
        if not code_str:
            print("  ⚠️  No custom_embedding_code produced by Agent 8.", flush=True)
            print("  → Attempting focused code-generation retry …", flush=True)
            code_str, default_params = self._retry_code_generation(spec)

        if not code_str:
            print("  ❌ Could not obtain executable code — skipping implementation.", flush=True)
            result["error"] = "No executable code produced"
            return result

        # Save the algorithm code
        code_path = os.path.join(self.output_dir, "novel_algorithm.py")
        with open(code_path, "w") as f:
            f.write(f"# Novel Algorithm: {algo_name}\n")
            f.write("# Auto-generated by the Novel Algorithm Pipeline\n\n")
            f.write(code_str)
        print(f"  → Code saved: {code_path}", flush=True)

        # --- 2. Test on small sample ---
        print("  [1/4] Validating generated code on small sample …", flush=True)
        test_ok, test_msg = self._test_custom_code(code_str, data, labels, default_params)
        if not test_ok:
            print(f"  ⚠️  Code validation failed: {test_msg}", flush=True)
            print("  → Attempting error recovery via Agent 10 …", flush=True)
            fixed_code, fixed_params, recovery_ok = self._attempt_error_recovery(
                code_str, test_msg, data, labels, default_params,
            )
            if recovery_ok:
                code_str = fixed_code
                default_params = fixed_params
                print("  ✅ Agent 10 fix validated — proceeding with corrected code", flush=True)
                result["code_validation"] = {"ok": True, "recovered_by_agent10": True}
                result["used_fallback"] = False
                # Save recovered code
                rec_path = os.path.join(self.output_dir, "novel_algorithm_recovered.py")
                with open(rec_path, "w") as f:
                    f.write(f"# Recovered by Agent 10\n# Original error: {test_msg[:200]}\n\n")
                    f.write(code_str)
                print(f"  → Recovered code saved: {rec_path}", flush=True)
            else:
                print("  ❌ Agent 10 could not fix the code — falling back to t-SNE", flush=True)
                result["code_validation"] = {"ok": False, "message": test_msg, "recovery_failed": True}
                embedding = self.dr_engine.compute_embedding(data, method="tsne")
                result["used_fallback"] = True
        else:
            print("  ✅ Code validated successfully", flush=True)
            result["code_validation"] = {"ok": True}
            result["used_fallback"] = False

        # Track the final executed code + params for downstream consumers
        # (e.g. the math/pseudocode layer needs the post-recovery version).
        if not result.get("used_fallback"):
            result["code"] = code_str
            result["params"] = default_params

        # --- 3. Iterative optimisation loop (or single-shot if disabled) ---
        if not result.get("used_fallback"):
            if self.config.convergence.enable_iterative_optimization:
                print("  [2/4] Starting iterative optimisation loop …", flush=True)
                embedding = self._iterative_optimise(
                    data, labels, label_names, code_str, default_params, algo_name
                )
            else:
                print("  [2/4] Iterative optimisation disabled — single-shot embedding …", flush=True)
                embedding = self._single_shot_embed(
                    data, labels, label_names, code_str, default_params, algo_name
                )
        else:
            # Fallback: just compute metrics once for the t-SNE embedding
            metrics = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self._metrics_k, lof_neighbors=self._lof_neighbors,
            )
            snap = self._extract_metrics_snapshot(metrics)
            self._print_metrics_snapshot(snap)

        # --- 4. Compare with baselines ---
        print("  [3/4] Comparing with baseline methods …", flush=True)
        if self.best_result:
            best_metrics = self.best_result["metrics"]
        else:
            # No iteration produced a best result (e.g. every iteration failed):
            # evaluate the final embedding now so the comparison and downstream
            # steps still have metrics instead of crashing on an unbound name.
            _m = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self._metrics_k, lof_neighbors=self._lof_neighbors,
            )
            best_metrics = self._extract_metrics_snapshot(_m)
        comparison = self._compare_with_baselines(
            data, labels, label_names, embedding, best_metrics, algo_name
        )
        result["comparison"] = comparison
        self._save_json("novel_comparison.json", comparison)

        # Distortion heatmap + full forensics are now generated in Phase 7
        print("  [4/4] Distortion analysis deferred to Phase 7 — Visual Forensics", flush=True)

        result["success"] = True
        result["optimization_history"] = self.optimization_history
        result["best_iteration"] = self.best_result["iteration"] if self.best_result else None

        # --- 5. Beat-target check against Phase 0 baseline_profile ---
        profile = self.context.get("baseline_profile")
        if profile and best_metrics:
            # best_metrics may be the nested dict from compute_all — flatten first
            novel_flat = (
                best_metrics
                if "Trustworthiness" in best_metrics
                else self._extract_metrics_snapshot(best_metrics)
            )
            targets = profile.get("beat_targets", {})
            tol = float(targets.get("tolerance", 0.0))
            failed = []
            def _cmp_ge(name, cur, tgt):
                if cur is None or tgt is None:
                    return
                if float(cur) < float(tgt) - tol:
                    failed.append(f"{name}: novel={float(cur):.4f} < baseline_winner={float(tgt):.4f} (tol={tol})")
            def _cmp_le(name, cur, tgt):
                if cur is None or tgt is None:
                    return
                if float(cur) > float(tgt) + tol:
                    failed.append(f"{name}: novel={float(cur):.4f} > baseline_winner={float(tgt):.4f} (tol={tol})")
            _cmp_ge("Trustworthiness",     novel_flat.get("Trustworthiness"),     targets.get("Trustworthiness"))
            _cmp_ge("Continuity",          novel_flat.get("Continuity"),          targets.get("Continuity"))
            _cmp_ge("Trustworthiness_AUC", novel_flat.get("Trustworthiness_AUC"), targets.get("Trustworthiness_AUC"))
            _cmp_ge("Continuity_AUC",      novel_flat.get("Continuity_AUC"),      targets.get("Continuity_AUC"))
            _cmp_le("Stress",              novel_flat.get("Stress"),              targets.get("Stress"))
            _cmp_ge("Spearman",            novel_flat.get("Spearman"),            targets.get("Spearman"))
            regression = len(failed) > 0
            tracked_keys = [
                "Trustworthiness", "Continuity",
                "Trustworthiness_AUC", "Continuity_AUC",
                "Stress", "Spearman",
            ]
            result["beat_target_check"] = {
                "winner_baseline": profile.get("winner", {}).get("name"),
                "targets":   {k: targets.get(k) for k in tracked_keys},
                "novel":     {k: novel_flat.get(k) for k in tracked_keys},
                "failed_metrics": failed,
                "regression_flag": regression,
            }
            if regression:
                print(f"\n  ⚠️  REGRESSION FLAG: novel did not beat baseline winner "
                      f"({profile.get('winner', {}).get('name')}) on:", flush=True)
                for f_msg in failed:
                    print(f"     - {f_msg}", flush=True)
            else:
                print(f"\n  ✅ Novel beats baseline winner "
                      f"({profile.get('winner', {}).get('name')}) on all tracked metrics.", flush=True)

        print(f"\n  🎉 Implementation & Evaluation complete for: {algo_name}", flush=True)
        return result

    # ------------------------------------------------------------------
    # Phase 7 — Visual Forensics (matches standard planner Phase 6)
    # ------------------------------------------------------------------

    def _phase7_visual_forensics(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        """
        Post-optimisation visual forensics analysis for novel algorithms.
        Mirrors the standard planner's Phase 6:
        1. Compute pointwise distortion metrics on the best embedding
        2. Cluster high-distortion regions (DBSCAN)
        3. Compute per-class distortion stats
        4. Compute hierarchy diff (dendrograms)
        5. Run Agent 9 — Visual Forensics Evaluator
        6. Generate all 5 forensic plots
        """
        if os.environ.get("DR_SKIP_BASELINES_AND_FORENSICS") == "1":
            print("  ⏭  Visual forensics skipped (DR_SKIP_BASELINES_AND_FORENSICS=1).", flush=True)
            return {"skipped": True}
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
        distortion_clusters = self._find_distortion_clusters(
            embedding, composite, threshold_pct=90,
        )
        print(f"  → Found {len(distortion_clusters)} high-distortion clusters", flush=True)

        # --- 3. Compute per-class distortion stats ---
        class_centroids: Dict[str, Any] = {}
        class_distortion_stats: Dict[str, Any] = {}
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

        def _centroid(mask_arr):
            if mask_arr.any():
                return float(embedding[mask_arr, 0].mean()), float(embedding[mask_arr, 1].mean())
            return 0.0, 0.0

        worst_dist_cx, worst_dist_cy = _centroid(worst_10_mask)
        worst_trust_cx, worst_trust_cy = _centroid(worst_trust_mask)
        worst_cont_cx, worst_cont_cy = _centroid(worst_cont_mask)

        # --- 5. Compute hierarchy diff for forensics ---
        print("  Computing hierarchy comparison (HD vs 2D) …", flush=True)
        best_dendrograms = self.best_result.get("dendrograms", {})
        # If dendrograms don't have linkage data, recompute
        if not best_dendrograms or best_dendrograms.get("hierarchy_hd", {}).get("linkage") is None:
            try:
                best_dendrograms = self.clustering_engine.compute_dendrograms(
                    hd_data=data, ld_data=embedding, labels=labels,
                )
            except Exception as exc:
                print(f"  ⚠️  Dendrogram computation failed: {exc}", flush=True)
                best_dendrograms = {}

        hierarchy_diff: Dict[str, Any] = {}
        if best_dendrograms:
            try:
                algo_label = getattr(self, '_novel_algo_name', 'Novel Algorithm')
                hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
                    best_dendrograms, label_names=label_names,
                    dr_method=algo_label,
                )
                rf_dist = hierarchy_diff.get("robinson_foulds_distance", "?")
                coph_corr = hierarchy_diff.get("cophenetic_correlation", "?")
                n_topo = len(hierarchy_diff.get("topology_changes", []))
                print(
                    f"  → Hierarchy: RF={rf_dist}, cophenetic_corr={coph_corr}, "
                    f"topology_changes={n_topo}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  ⚠️  Hierarchy diff failed: {exc}", flush=True)

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
        try:
            vi_result = self.agent9.run(agent_ctx)
            p = self._save_json("agent9_visual_forensics.json", vi_result)
            print(f"  → Saved: {p}", flush=True)
        except Exception as exc:
            print(f"  ⚠️  Agent 9 failed: {exc}", flush=True)
            vi_result = {}

        # --- 8. Generate forensic plots ---
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

        # Highlighted areas from Agent 9
        highlighted_regions = vi_result.get("highlighted_regions", [])
        p2 = None
        if highlighted_regions:
            print(f"  Generating highlighted areas plot ({len(highlighted_regions)} regions) …", flush=True)
            try:
                p2 = plot_highlighted_areas(
                    embedding=embedding,
                    labels=labels,
                    label_names=label_names,
                    highlighted_regions=highlighted_regions,
                    composite_distortion=composite,
                    output_dir=self.output_dir,
                )
                print(f"  → Saved: {p2}", flush=True)
            except Exception as exc:
                print(f"  ⚠️  Highlighted areas plot failed: {exc}", flush=True)
        else:
            print("  ⚠️  Agent 9 returned no highlighted regions.", flush=True)

        # Per-class distortion summary
        p3 = None
        if labels is not None:
            print("  Generating per-class distortion summary …", flush=True)
            try:
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
            except Exception as exc:
                print(f"  ⚠️  Class distortion plot failed: {exc}", flush=True)

        # Dendrogram comparison
        p4 = None
        if best_dendrograms:
            print("  Generating dendrogram comparison plots …", flush=True)
            try:
                p4 = plot_dendrogram_comparison(
                    dendrograms=best_dendrograms,
                    hierarchy_diff=hierarchy_diff,
                    label_names=label_names,
                    output_dir=self.output_dir,
                )
                if p4:
                    print(f"  → Saved: {p4}", flush=True)
            except Exception as exc:
                print(f"  ⚠️  Dendrogram comparison plot failed: {exc}", flush=True)

        # Hierarchy change detail (arrows on 2D scatter + merge-rank chart)
        p5 = None
        if best_dendrograms and hierarchy_diff:
            try:
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
            except Exception as exc:
                print(f"  ⚠️  Hierarchy change detail plot failed: {exc}", flush=True)

        print("  ✅ Visual forensics complete.", flush=True)

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
        Identify spatial clusters of high-distortion points using DBSCAN.
        Returns list of dicts with center_x, center_y, radius, n_points, mean_distortion.
        """
        from sklearn.cluster import DBSCAN

        threshold = np.percentile(composite_distortion, threshold_pct)
        high_mask = composite_distortion >= threshold
        if high_mask.sum() < 3:
            return []

        high_points = embedding[high_mask]
        high_dist = composite_distortion[high_mask]

        # Set eps relative to embedding spread
        x_range = embedding[:, 0].max() - embedding[:, 0].min()
        y_range = embedding[:, 1].max() - embedding[:, 1].min()
        eps = max(x_range, y_range) * 0.08

        db = DBSCAN(eps=eps, min_samples=max(3, int(high_mask.sum() * 0.05)))
        cluster_labels = db.fit_predict(high_points)

        clusters = []
        for cl in sorted(set(cluster_labels)):
            if cl == -1:
                continue
            cmask = cluster_labels == cl
            pts = high_points[cmask]
            dists = high_dist[cmask]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
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

        clusters.sort(key=lambda c: -c["mean_distortion"])
        return clusters[:max_clusters]

    # ------------------------------------------------------------------
    # Single-shot embedding (used when iterative optimization is disabled)
    # ------------------------------------------------------------------

    def _single_shot_embed(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        code_str: str,
        default_params: Dict[str, Any],
        algo_name: str,
    ) -> np.ndarray:
        """Run the novel algorithm exactly once with default params.

        Skips Agent 6 (hyperparam optimization) and Agent 7 (quality scoring)
        entirely. Populates self.best_result so downstream baseline comparison,
        forensics, and the UI iter_*_metrics.json file all keep working.
        """
        hyperparams = copy.deepcopy(default_params)
        fallback_method = "tsne"

        print(f"  ℹ️  Using novel algorithm: {algo_name} (single-shot mode)", flush=True)

        # 1. Compute embedding (with one recovery attempt on failure)
        emb, exec_ok, exec_msg = self._run_custom_code(
            code_str, data, labels, hyperparams
        )
        if not exec_ok:
            print(f"  ⚠️  Custom code failed: {exec_msg}", flush=True)
            print("  → Attempting error recovery via Agent 10 …", flush=True)
            fixed_code, fixed_params, recovery_ok = self._attempt_error_recovery(
                code_str, exec_msg, data, labels, hyperparams,
            )
            if recovery_ok:
                code_str = fixed_code
                hyperparams = fixed_params
                emb, exec_ok2, exec_msg2 = self._run_custom_code(
                    code_str, data, labels, hyperparams
                )
                if not exec_ok2:
                    print(f"  → Falling back to standard {fallback_method}", flush=True)
                    emb = self.dr_engine.compute_embedding(data, method=fallback_method)
            else:
                print(f"  → Falling back to standard {fallback_method}", flush=True)
                emb = self.dr_engine.compute_embedding(data, method=fallback_method)

        embedding = emb

        # 2. Metrics + dendrograms (needed for downstream phases)
        metrics = self.metrics_engine.compute_all(
            hd_data=data, ld_data=embedding, labels=labels,
            k=self._metrics_k, lof_neighbors=self._lof_neighbors,
        )
        snap = self._extract_metrics_snapshot(metrics)
        self._print_metrics_snapshot(snap)

        dendrograms = self.clustering_engine.compute_dendrograms(
            hd_data=data, ld_data=embedding, labels=labels,
        )
        hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
            dendrograms, label_names=label_names, dr_method=algo_name,
        )

        # 3. Render plot for the UI
        plot_path = os.path.join(self.output_dir, "iter_001.png")
        self.dr_engine.render_plot(
            embedding, labels=labels, label_names=label_names,
            title=f"{algo_name} — single-shot",
            quality_score=None,
            save_path=plot_path,
        )

        # 4. Record a single history entry so UI/forensics see something
        record = {
            "iteration": 1,
            "algorithm": algo_name,
            "hyperparameters": copy.deepcopy(hyperparams),
            "implicit_score": None,
            "evaluator_score": None,
            "convergence_score": None,
            "code_updated": False,
            "metrics_snapshot": snap,
            "iterative_optimization_disabled": True,
            "hierarchy_rf_distance": hierarchy_diff.get("robinson_foulds_distance"),
            "hierarchy_cophenetic_corr": hierarchy_diff.get("cophenetic_correlation"),
            "hierarchy_n_topology_changes": len(hierarchy_diff.get("topology_changes", [])),
        }
        self.optimization_history.append(record)
        self.context["optimization_history"] = self.optimization_history

        with open(os.path.join(self.output_dir, "iter_1_metrics.json"), "w") as _f:
            json.dump({
                "iteration": 1,
                "metrics": snap,
                "implicit_score": None,
                "evaluator_score": None,
                "iterative_optimization_disabled": True,
            }, _f, indent=2, default=str)

        # 5. Populate best_result (downstream code reads from here)
        self.best_result = {
            "iteration": 1,
            "hyperparameters": hyperparams,
            "algorithm_code": code_str,
            "embedding": embedding,
            "metrics": metrics,
            "eval_result": {},
            "opt_result": {},
            "plot_path": plot_path,
        }

        return embedding

    # ------------------------------------------------------------------
    # Embedding-collapse guardrail
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_embedding_collapse(
        embedding: np.ndarray,
        snap: Dict[str, Any],
        prev_best_snap: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Detect degenerate embeddings (mode collapse, point pile-up, exploded MDR).

        Two layers:
          (1) Geometric — independent of metric values; spots pile-ups and
              MDR explosions even when scoring metrics happen to look good.
          (2) Implausibility — saturated Silhouette while neighborhood-
              preservation metrics regress is the textbook collapse signature.

        Returns ``(collapsed: bool, reasons: list[str])``.
        """
        reasons: List[str] = []

        try:
            sil = float(snap.get("Silhouette") or 0.0)
            T   = float(snap.get("Trustworthiness") or 0.0)
            C   = float(snap.get("Continuity") or 0.0)
            mdr = float(snap.get("Mean_Distance_Ratio") or 0.0)
            lof = float(snap.get("LOF_Outlier_pct") or 0.0)
        except (TypeError, ValueError):
            sil = T = C = mdr = lof = 0.0

        # ---- (1) Geometric: per-axis pile-up + MDR ----
        try:
            if embedding is not None and embedding.ndim == 2 and embedding.shape[0] > 10:
                for axis in range(embedding.shape[1]):
                    col = embedding[:, axis]
                    full = float(col.max() - col.min())
                    if full <= 0.0:
                        reasons.append(f"axis {axis} range is zero — total collapse")
                        continue
                    p5, p95 = np.percentile(col, [5.0, 95.0])
                    bulk = float(p95 - p5)
                    if full > 0 and bulk / full < 1e-3:
                        reasons.append(
                            f"axis {axis} bulk-spread (p5–p95)={bulk:.3e} is "
                            f"<0.1% of full range {full:.3e} — pile-up + outliers"
                        )
                # effective unique-point fraction (rounded to ε of axis range)
                ranges = embedding.max(axis=0) - embedding.min(axis=0)
                ranges = np.where(ranges > 0, ranges, 1.0)
                eps = ranges * 1e-3
                quantized = np.round(embedding / eps).astype(np.int64)
                unique_pts = len({tuple(row) for row in quantized})
                if unique_pts < max(5, int(0.05 * embedding.shape[0])):
                    reasons.append(
                        f"only {unique_pts} unique embedded locations for "
                        f"{embedding.shape[0]} points — mass pile-up onto a "
                        "few centroids"
                    )
        except Exception:
            pass

        if mdr >= 1e6:
            reasons.append(
                f"Mean_Distance_Ratio={mdr:.3e} — LD distance scale degenerate"
            )

        # ---- (2) Implausibility: Silhouette saturated while T/C broken ----
        if sil >= 0.95 and (T < 0.85 or C < 0.85):
            reasons.append(
                f"Silhouette={sil:.4f} saturated while T={T:.3f}/C={C:.3f} "
                "<0.85 — implausible without cluster collapse"
            )

        # T/C regression vs the previous best (>5pp) alongside silhouette gain
        if prev_best_snap and sil >= 0.85:
            try:
                prev_T = float(prev_best_snap.get("Trustworthiness") or 0.0)
                prev_C = float(prev_best_snap.get("Continuity") or 0.0)
                if (prev_T - T) > 0.05 or (prev_C - C) > 0.05:
                    reasons.append(
                        f"T/C regressed >5pp vs previous best "
                        f"(T:{prev_T:.3f}→{T:.3f}, C:{prev_C:.3f}→{C:.3f}) "
                        f"while Silhouette={sil:.3f} — chasing one metric at "
                        "the cost of faithful representation"
                    )
            except (TypeError, ValueError):
                pass

        if lof >= 25.0:
            reasons.append(
                f"LOF_Outlier_pct={lof:.2f}% — long outlier tail vs the bulk"
            )

        return (len(reasons) > 0), reasons

    # ------------------------------------------------------------------
    # Iterative optimisation sub-loop
    # ------------------------------------------------------------------

    def _iterative_optimise(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        code_str: str,
        default_params: Dict[str, Any],
        algo_name: str,
    ) -> np.ndarray:
        """Run the novel algorithm iteratively, using Agent 6 + Agent 7."""
        conv = self.config.convergence
        fallback_method = "tsne"            # generic fallback, independent of user selection
        hyperparams = copy.deepcopy(default_params)
        current_code = code_str

        best_score = -1.0
        patience_counter = 0
        embedding = None

        print(f"  ℹ️  Using novel algorithm: {algo_name}", flush=True)
        print(f"  ℹ️  Max iterations: {conv.max_iterations}, target: {conv.quality_score_target}", flush=True)

        for iteration in range(1, conv.max_iterations + 1):
            print(f"\n  [{iteration}/{conv.max_iterations}] Computing embedding …", flush=True)

            # --- 1. Compute embedding ---
            emb, exec_ok, exec_msg = self._run_custom_code(
                current_code, data, labels, hyperparams
            )
            if not exec_ok:
                print(f"  ⚠️  Custom code failed: {exec_msg}", flush=True)
                print("  → Attempting error recovery via Agent 10 …", flush=True)
                fixed_code, fixed_params, recovery_ok = self._attempt_error_recovery(
                    current_code, exec_msg, data, labels, hyperparams,
                )
                recovered_emb = None
                if recovery_ok:
                    candidate_code = fixed_code
                    candidate_params = fixed_params
                    emb2, exec_ok2, exec_msg2 = self._run_custom_code(
                        candidate_code, data, labels, candidate_params
                    )
                    if exec_ok2:
                        print("  ✅ Agent 10 fix succeeded on full data", flush=True)
                        current_code = candidate_code
                        hyperparams = candidate_params
                        cp = os.path.join(self.output_dir, f"recovered_iter{iteration:03d}.py")
                        with open(cp, "w") as f:
                            f.write(f"# Recovered by Agent 10 at iteration {iteration}\n\n")
                            f.write(current_code)
                        recovered_emb = emb2
                    else:
                        print(f"  ⚠️  Agent 10 fix still fails on full data: {exec_msg2}", flush=True)
                else:
                    print("  ⚠️  Agent 10 could not produce a fix.", flush=True)

                # No t-SNE fallback. Keep the previous best if any; otherwise
                # abort the experiment — we don't want another t-SNE result.
                if recovered_emb is None:
                    if self.best_result is not None:
                        prev_iter = self.best_result.get("iteration")
                        print(
                            f"  ⏹ Iteration {iteration}: novel code is broken and "
                            "Agent 10 could not recover. Skipping t-SNE fallback "
                            f"and keeping the previous best embedding from "
                            f"iteration {prev_iter}. Stopping the iterative loop.",
                            flush=True,
                        )
                        break
                    raise RuntimeError(
                        f"Iteration {iteration} produced no valid embedding and "
                        "no prior iteration succeeded. Aborting experiment — "
                        "the t-SNE fallback is disabled by user policy."
                    )
                emb = recovered_emb
            embedding = emb

            # --- 2. Compute metrics ---
            metrics = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self._metrics_k, lof_neighbors=self._lof_neighbors,
            )
            snap = self._extract_metrics_snapshot(metrics)

            # --- 2b. Collapse / implausibility guardrail ---
            prev_best_snap = (
                (self.best_result or {}).get("metrics_snapshot")
                if isinstance(self.best_result, dict) else None
            )
            if prev_best_snap is None and self.optimization_history:
                # walk back to the most recent non-collapsed iteration
                for h in reversed(self.optimization_history):
                    if not h.get("collapsed") and h.get("metrics_snapshot"):
                        prev_best_snap = h.get("metrics_snapshot")
                        break
            collapsed, collapse_reasons = self._detect_embedding_collapse(
                embedding, snap, prev_best_snap=prev_best_snap,
            )
            if collapsed:
                print(
                    "  🛑 Collapse guardrail tripped — this iteration will be "
                    "quarantined and CANNOT become best.",
                    flush=True,
                )
                for r in collapse_reasons:
                    print(f"     · {r}", flush=True)

            # --- 3. Compute dendrograms + hierarchy diff ---
            dendrograms = self.clustering_engine.compute_dendrograms(
                hd_data=data, ld_data=embedding, labels=labels,
            )
            hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
                dendrograms, label_names=label_names, dr_method=algo_name,
            )

            # --- 4. Build iteration context for agents ---
            plot_path = os.path.join(self.output_dir, f"iter_{iteration:03d}.png")
            iter_ctx = {
                **self.context,
                "current_iteration": iteration,
                "dr_method": algo_name,
                "current_hyperparameters": hyperparams,
                "current_algorithm_code": current_code,
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
                "collapse_warning": {
                    "tripped": collapsed,
                    "reasons": collapse_reasons,
                } if collapsed else None,
            }

            # --- 5. Agent 7 — Independent evaluation ---
            eval_result = self.agent7_eval.run(iter_ctx)
            quality_score = float(
                eval_result.get("overall_assessment", {}).get("quality_score", 0)
            )
            self._save_json(f"iter_{iteration:03d}_agent7_evaluation.json", eval_result)

            # --- 6. Agent 6 — Hyperparameter optimization ---
            opt_result = self.agent6.run(iter_ctx)
            implicit_score = float(opt_result.get("quality_score", 0))
            self._save_json(f"iter_{iteration:03d}_agent6_optimization.json", opt_result)

            # Convergence score blends both LLM scorers so the loop benefits
            # from the optimizer's recommendation strength AND the independent
            # evaluator's verdict on the rendered embedding. Both are reported
            # on the same 0-1 quality scale by their respective JSON schemas.
            convergence_score = 0.5 * implicit_score + 0.5 * quality_score

            # Collapsed iterations cannot win the "best" race regardless of
            # what the LLM scorers said. Force the convergence score to 0 so
            # later code paths (best update, target-reached) ignore it. The
            # raw implicit/evaluator scores stay in the record for forensics.
            if collapsed:
                convergence_score = 0.0

            # --- 7. Render plot ---
            # Defer plot rendering to new-best (or target-hit) iterations only.
            # Per-iteration plots aren't consumed by Agents 6/7, so rendering
            # losing iterations just burns matplotlib I/O. The final best
            # embedding is guaranteed a plot via the post-loop fallback below.
            will_be_new_best = (not collapsed) and convergence_score > best_score
            will_hit_target = (
                (not collapsed)
                and convergence_score >= conv.quality_score_target
            )
            if will_be_new_best or will_hit_target:
                self.dr_engine.render_plot(
                    embedding, labels=labels, label_names=label_names,
                    title=f"{algo_name} — Iteration {iteration}",
                    quality_score=implicit_score,
                    save_path=plot_path,
                )
            else:
                plot_path = None  # so history record reflects "not rendered"

            # Persist the LD embedding for this iteration so downstream
            # metric scripts (e.g. k-NN accuracy) can
            # re-evaluate metrics in-place without re-running the iterative
            # loop. Cheap: a 5000×2 float64 array is ~80 KB; bigger datasets
            # cap at a few MB per iteration.
            try:
                np.save(
                    os.path.join(self.output_dir, f"iter_{iteration:03d}.npy"),
                    embedding,
                )
            except Exception as _exc:
                print(f"  ⚠️  could not persist iter_{iteration:03d}.npy: {_exc}",
                      flush=True)

            # --- 8. Record history ---
            record = {
                "iteration": iteration,
                "algorithm": algo_name,
                "hyperparameters": copy.deepcopy(hyperparams),
                "implicit_score": implicit_score,
                "evaluator_score": quality_score,
                "convergence_score": convergence_score,
                "code_updated": bool(opt_result.get("code_update")),
                "metrics_snapshot": snap,
                "agent6_score_rationale": opt_result.get("score_rationale", ""),
                "agent6_all_recommendations": opt_result.get("recommendations", []),
                "agent6_key_weaknesses": opt_result.get("overall_assessment", {}).get("key_weaknesses", []),
                "agent7_key_strengths": eval_result.get("overall_assessment", {}).get("key_strengths", []),
                "agent7_key_weaknesses": eval_result.get("overall_assessment", {}).get("key_weaknesses", []),
                "agent7_improvement_priorities": eval_result.get("improvement_priorities", [])[:3],
                "hierarchy_rf_distance": hierarchy_diff.get("robinson_foulds_distance"),
                "hierarchy_cophenetic_corr": hierarchy_diff.get("cophenetic_correlation"),
                "hierarchy_n_topology_changes": len(hierarchy_diff.get("topology_changes", [])),
                "collapsed": bool(collapsed),
                "collapse_reasons": collapse_reasons if collapsed else [],
            }
            self.optimization_history.append(record)

            # Write per-iteration metrics for the UI
            iter_metrics_path = os.path.join(
                self.output_dir, f"iter_{iteration}_metrics.json"
            )
            with open(iter_metrics_path, "w") as _f:
                json.dump({
                    "iteration": iteration,
                    "metrics": snap,
                    "implicit_score": implicit_score,
                    "evaluator_score": quality_score,
                    "collapsed": bool(collapsed),
                    "collapse_reasons": collapse_reasons if collapsed else [],
                }, _f, indent=2, default=str)

            print(
                f"  → implicit={implicit_score:.2f}  "
                f"evaluator={quality_score:.2f}  "
                f"convergence={convergence_score:.2f}  "
                f"best={best_score:.2f}",
                flush=True,
            )
            self._print_metrics_snapshot(snap)

            # --- 9. Convergence check ---
            # Collapsed iters cannot satisfy the target nor become best.
            if not collapsed and convergence_score >= conv.quality_score_target:
                print(f"  ✅ Target quality {conv.quality_score_target} reached.", flush=True)
                self.best_result = {
                    "iteration": iteration,
                    "hyperparameters": hyperparams,
                    "algorithm_code": current_code,
                    "embedding": embedding,
                    "metrics": metrics,
                    "metrics_snapshot": snap,
                    "eval_result": eval_result,
                    "opt_result": opt_result,
                    "plot_path": plot_path,
                }
                break

            if (not collapsed) and convergence_score > best_score:
                best_score = convergence_score
                patience_counter = 0
                self.best_result = {
                    "iteration": iteration,
                    "hyperparameters": copy.deepcopy(hyperparams),
                    "algorithm_code": current_code,
                    "embedding": embedding,
                    "metrics": metrics,
                    "metrics_snapshot": snap,
                    "eval_result": eval_result,
                    "opt_result": opt_result,
                    "plot_path": plot_path,
                }
            else:
                patience_counter += 1

            if patience_counter >= conv.patience:
                print(f"  ⏹ No improvement for {conv.patience} iterations — stopping.", flush=True)
                break

            # --- 10. Apply Agent 6 recommendations ---
            old_params = copy.deepcopy(hyperparams)
            hyperparams, applied_recs, skipped_recs = self._apply_recommendations(
                hyperparams, opt_result
            )

            params_delta = {}
            for k in set(list(old_params.keys()) + list(hyperparams.keys())):
                if old_params.get(k) != hyperparams.get(k):
                    params_delta[k] = {"from": old_params.get(k), "to": hyperparams.get(k)}

            if self.optimization_history:
                self.optimization_history[-1]["params_delta"] = params_delta
                self.optimization_history[-1]["applied_recommendations"] = applied_recs
                self.optimization_history[-1]["skipped_recommendations"] = skipped_recs

            # Check for code update from Agent 6
            code_update = opt_result.get("code_update")
            code_was_updated = False
            if code_update and isinstance(code_update, str) and code_update.strip():
                print("  → Agent 6 proposed code update — validating …", flush=True)
                test_ok, test_msg = self._test_custom_code(code_update, data, labels, hyperparams)
                if test_ok:
                    current_code = code_update
                    code_was_updated = True
                    cp = os.path.join(self.output_dir, f"custom_algorithm_iter{iteration:03d}.py")
                    with open(cp, "w") as f:
                        f.write(f"# Updated by Agent 6 at iteration {iteration}\n\n")
                        f.write(code_update)
                    print(f"  ✅ Code update applied → {cp}", flush=True)
                else:
                    print(f"  ⚠️  Code update failed: {test_msg} — trying Agent 10 …", flush=True)
                    fix_code, fix_params, fix_ok = self._attempt_error_recovery(
                        code_update, test_msg, data, labels, hyperparams,
                    )
                    if fix_ok:
                        current_code = fix_code
                        hyperparams = fix_params
                        code_was_updated = True
                        cp = os.path.join(self.output_dir, f"custom_algorithm_iter{iteration:03d}_recovered.py")
                        with open(cp, "w") as f:
                            f.write(f"# Agent 6 update recovered by Agent 10 at iteration {iteration}\n\n")
                            f.write(fix_code)
                        print(f"  ✅ Agent 10 fixed Agent 6 code update → {cp}", flush=True)
                    else:
                        print("  ⚠️  Agent 10 could not fix the code update — keeping previous code.", flush=True)

            if not params_delta and not code_was_updated:
                print("  ⚠️  No parameter or code changes this iteration.", flush=True)
                if self.optimization_history:
                    self.optimization_history[-1]["stale_iteration"] = True

            if params_delta:
                changes = ", ".join(f"{k}: {v['from']}→{v['to']}" for k, v in params_delta.items())
                print(f"  → Changes: {changes}", flush=True)

        # Save iteration history
        hist_path = os.path.join(self.output_dir, "optimization_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.optimization_history, f, indent=2, default=str)

        # Save final algorithm code
        if current_code:
            final_path = os.path.join(self.output_dir, "novel_algorithm_final.py")
            with open(final_path, "w") as f:
                f.write(f"# Final novel algorithm after {len(self.optimization_history)} iterations\n\n")
                f.write(current_code)
            print(f"  → Final algorithm code saved: {final_path}", flush=True)

        # Convergence plot
        if self.optimization_history:
            p = plot_convergence_curve(
                self.optimization_history,
                self.output_dir,
                quality_target=conv.quality_score_target,
            )
            print(f"  → Convergence plot saved: {p}", flush=True)

        # Use best embedding if available
        if self.best_result:
            embedding = self.best_result["embedding"]
            # Persist the final / best LD embedding under a stable filename
            # so downstream metric scripts (k-NN accuracy, stress, etc.)
            # can re-evaluate without re-running the iterative loop
            # or guessing which iter_NNN.npy is the "best" one.
            try:
                np.save(
                    os.path.join(self.output_dir, "best_embedding.npy"),
                    embedding,
                )
                print(
                    f"  → Best embedding saved: {os.path.join(self.output_dir, 'best_embedding.npy')}",
                    flush=True,
                )
            except Exception as _exc:
                print(f"  ⚠️  could not persist best_embedding.npy: {_exc}",
                      flush=True)

        return embedding

    # ------------------------------------------------------------------
    # Metrics extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics_snapshot(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metrics into a simple snapshot dict for the UI."""
        global_m = metrics.get("Global Structure & HD<->2D Correlation", {})
        local_m = metrics.get("Neighborhood Preservation", {})
        cluster_m = metrics.get("Clustering & Label Agreement", {})
        lof_m = metrics.get("Outlier Detection", {})
        return {
            "Spearman": global_m.get("Spearman Correlation"),
            "Stress": global_m.get("Stress"),
            "Mean_Distance_Ratio": global_m.get("Mean Distance Ratio"),
            "Trustworthiness": next(
                (v for k, v in local_m.items()
                 if k.startswith("Trustworthiness") and "AUC" not in k and "curve" not in k),
                None,
            ),
            "Continuity": next(
                (v for k, v in local_m.items()
                 if k.startswith("Continuity") and "AUC" not in k and "curve" not in k),
                None,
            ),
            "Trustworthiness_AUC": local_m.get("Trustworthiness AUC"),
            "Continuity_AUC": local_m.get("Continuity AUC"),
            "k_sweep": local_m.get("k_sweep"),
            "Trustworthiness_curve": local_m.get("Trustworthiness curve"),
            "Continuity_curve": local_m.get("Continuity curve"),
            "Silhouette": cluster_m.get("Silhouette Score"),
            "LOF_Median": lof_m.get("LOF Median"),
            "LOF_Outlier_pct": lof_m.get("LOF Outlier %"),
        }

    @staticmethod
    def _print_metrics_snapshot(snap: Dict[str, Any]) -> None:
        """Print key metrics in a compact line. AUC variants of T/C are
        preferred; the single-k values are shown only when the AUC sweep
        was unavailable (very small datasets)."""
        def _f(v):
            return f"{float(v):.4f}" if v is not None else "—"
        t_auc = snap.get("Trustworthiness_AUC")
        c_auc = snap.get("Continuity_AUC")
        if t_auc is not None or c_auc is not None:
            print(
                f"  → T_AUC={_f(t_auc)}  "
                f"C_AUC={_f(c_auc)}  "
                f"S={_f(snap.get('Stress'))}  "
                f"Sil={_f(snap.get('Silhouette'))}",
                flush=True,
            )
        else:
            print(
                f"  → T={_f(snap.get('Trustworthiness'))}  "
                f"C={_f(snap.get('Continuity'))}  "
                f"S={_f(snap.get('Stress'))}  "
                f"Sil={_f(snap.get('Silhouette'))}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Recommendation application (adapted from standard Planner)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_recommendations(
        current: Dict[str, Any], opt_result: Dict[str, Any]
    ) -> tuple:
        """Parse Agent 6 recommendations and apply parameter updates."""
        updated = copy.deepcopy(current)
        recs = opt_result.get("recommendations", [])
        applied, skipped = [], []

        for rec in recs:
            param = rec.get("parameter", "")
            suggested = rec.get("suggested_value")
            if not param or suggested is None:
                continue
            try:
                if param in updated:
                    if isinstance(updated[param], int):
                        updated[param] = int(float(suggested))
                    elif isinstance(updated[param], float):
                        updated[param] = float(suggested)
                    else:
                        updated[param] = suggested
                else:
                    # New param — custom code may reference it
                    if isinstance(suggested, str):
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
                applied.append(rec)
            except (ValueError, TypeError):
                skipped.append(rec)

        return updated, applied, skipped

    # ------------------------------------------------------------------
    # Baseline comparison with metrics table
    # ------------------------------------------------------------------

    def _compare_with_baselines(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        novel_embedding: np.ndarray,
        novel_metrics: Dict[str, Any],
        algo_name: str,
        output_prefix: str = "novel",
        comparison_title: str = "Novel Algorithm vs Baselines",
        metrics_plot_title: str = "Metrics Comparison — Novel vs Baselines",
        metrics_json_label: str = "Novel Algorithm",
    ) -> Dict[str, Any]:
        """Run baseline methods and produce comparison plots + metrics table."""
        # Fast-path: skip baseline DR runs and all comparison plots, but still
        # persist the LLM-algorithm metrics so downstream aggregators work.
        if os.environ.get("DR_SKIP_BASELINES_AND_FORENSICS") == "1":
            METRIC_KEYS_FAST = [
                ("Trustworthiness_AUC", "↑", "Trust (AUC)"),
                ("Continuity_AUC",     "↑", "Cont (AUC)"),
                ("Spearman",           "↑", "Spearman"),
                ("Silhouette",         "↑", "Silhouette"),
                ("Stress",             "↓", "Stress"),
                ("LOF_Median",         "↓", "LOF"),
            ]
            novel_snap = self._extract_metrics_snapshot(novel_metrics)
            llm_row = {"method": f"★ {algo_name}",
                       **{k: novel_snap.get(k) for k, _, _ in METRIC_KEYS_FAST}}
            self._save_json(f"{output_prefix}_metrics.json", {
                "algorithm": algo_name,
                "algorithm_label": metrics_json_label,
                "used_fallback": False,
                "metrics_snapshot": novel_snap,
                "comparison_table": [llm_row],
                "baselines_skipped": True,
            })
            print("  ⏭  Baseline comparison skipped (DR_SKIP_BASELINES_AND_FORENSICS=1).", flush=True)
            return {"comparison_table": [llm_row], "comparison_plot": None,
                    "baselines_skipped": True}

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        baselines = ["tsne", "umap", "pacmap"]
        results = {}

        for method in baselines:
            try:
                emb = self.dr_engine.compute_embedding(data, method=method)
                m = self.metrics_engine.compute_all(
                    hd_data=data, ld_data=emb, labels=labels,
                    k=self._metrics_k, lof_neighbors=self._lof_neighbors,
                )
                results[method] = {"embedding": emb, "metrics": m}
            except Exception as exc:
                print(f"  ⚠️  Baseline {method} failed: {exc}", flush=True)
                results[method] = {"error": str(exc)}

        palette = [
            "#E74C3C", "#2E86AB", "#27AE60", "#F39C12", "#8E44AD",
            "#16A085", "#D35400", "#2C3E50", "#C0392B", "#1ABC9C",
        ]

        # ── 1. Side-by-side embedding comparison ──
        valid_baselines = [b for b in baselines if "embedding" in results.get(b, {})]
        n_cols = 1 + len(valid_baselines)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
        if n_cols == 1:
            axes = [axes]

        def _scatter(ax, emb, title_str, subtitle=""):
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

        def _metric_subtitle(m):
            s = self._extract_metrics_snapshot(m)
            parts = []
            t_auc = s.get("Trustworthiness_AUC")
            c_auc = s.get("Continuity_AUC")
            t_single = s.get("Trustworthiness")
            c_single = s.get("Continuity")
            t_val = t_auc if t_auc is not None else t_single
            c_val = c_auc if c_auc is not None else c_single
            t_label = "T_AUC" if t_auc is not None else "T"
            c_label = "C_AUC" if c_auc is not None else "C"
            if t_val is not None:
                parts.append(f"{t_label}={float(t_val):.3f}")
            if c_val is not None:
                parts.append(f"{c_label}={float(c_val):.3f}")
            if s.get("Stress") is not None:
                parts.append(f"S={float(s['Stress']):.3f}")
            return "  ".join(parts)

        _scatter(axes[0], novel_embedding, f"★ {algo_name}", _metric_subtitle(novel_metrics))
        for i, method in enumerate(valid_baselines):
            _scatter(axes[i + 1], results[method]["embedding"],
                     method.upper(), _metric_subtitle(results[method]["metrics"]))

        fig.suptitle(comparison_title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        cmp_path = os.path.join(self.output_dir, f"{output_prefix}_comparison.png")
        fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Comparison plot saved: {cmp_path}", flush=True)

        # ── 2. Build flat metrics comparison table ──
        # Trustworthiness/Continuity are reported via their k-sweep AUC so the
        # comparison reflects neighbourhood preservation across scales rather
        # than a single arbitrary k. Display labels keep the headers readable.
        METRIC_KEYS = [
            ("Trustworthiness_AUC", "↑", "Trust (AUC)"),
            ("Continuity_AUC",     "↑", "Cont (AUC)"),
            ("Spearman",           "↑", "Spearman"),
            ("Silhouette",         "↑", "Silhouette"),
            ("Stress",             "↓", "Stress"),
            ("LOF_Median",         "↓", "LOF"),
        ]
        novel_snap = self._extract_metrics_snapshot(novel_metrics)
        table_rows = [{"method": f"★ {algo_name}",
                       **{k: novel_snap.get(k) for k, _, _ in METRIC_KEYS}}]
        for method in valid_baselines:
            bsnap = self._extract_metrics_snapshot(results[method]["metrics"])
            table_rows.append({"method": method.upper(),
                               **{k: bsnap.get(k) for k, _, _ in METRIC_KEYS}})

        # Print table to log
        header = f"  {'Method':<22}" + "".join(f"{label+' '+d:>14}" for _, d, label in METRIC_KEYS)
        print(f"\n{header}", flush=True)
        print(f"  {'─' * (22 + 14 * len(METRIC_KEYS))}", flush=True)
        for row in table_rows:
            vals = "".join(
                f"{float(row[k]):>14.4f}" if row.get(k) is not None else f"{'—':>14}"
                for k, _, _ in METRIC_KEYS
            )
            print(f"  {row['method']:<22}{vals}", flush=True)

        # ── 3. Metrics comparison table as a visual plot ──
        self._plot_metrics_table(table_rows, METRIC_KEYS, algo_name, output_prefix=output_prefix)

        # ── 4. Metrics radar / bar chart ──
        self._plot_metrics_bars(table_rows, METRIC_KEYS, algo_name, output_prefix=output_prefix,
                    chart_title=metrics_plot_title)

        # Save JSON
        comparison_table = {"novel": {"name": algo_name, "metrics_snapshot": novel_snap}}
        for method in valid_baselines:
            bsnap = self._extract_metrics_snapshot(results[method]["metrics"])
            comparison_table[method] = {"name": method.upper(), "metrics_snapshot": bsnap}

        self._save_json(f"{output_prefix}_metrics.json", {
            "algorithm": algo_name,
            "algorithm_label": metrics_json_label,
            "used_fallback": False,
            "metrics_snapshot": novel_snap,
            "comparison_table": table_rows,
        })

        return {
            "comparison_table": table_rows,
            "comparison_plot": cmp_path,
        }

    def _plot_metrics_table(
        self,
        table_rows: List[Dict[str, Any]],
        metric_keys: List[tuple],
        algo_name: str,
        output_prefix: str = "novel",
    ) -> str:
        """Render the comparison table as a clean matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # metric_keys may be (key, dir) or (key, dir, label)
        def _unpack(t):
            return (t[0], t[1], t[2] if len(t) > 2 else t[0])

        methods = [r["method"] for r in table_rows]
        unpacked = [_unpack(t) for t in metric_keys]
        cols = [k for k, _, _ in unpacked]
        dirs = {k: d for k, d, _ in unpacked}

        cell_text = []
        cell_colors = []
        for row in table_rows:
            txt_row, clr_row = [], []
            for col in cols:
                v = row.get(col)
                if v is not None:
                    txt_row.append(f"{float(v):.4f}")
                else:
                    txt_row.append("—")
                    clr_row.append("#ffffff")
                    continue
                # Determine best value for this column
                col_vals = [float(r[col]) for r in table_rows if r.get(col) is not None]
                if dirs[col] == "↑":
                    is_best = float(v) >= max(col_vals) - 1e-6
                else:
                    is_best = float(v) <= min(col_vals) + 1e-6
                clr_row.append("#d4edda" if is_best else "#ffffff")
            cell_text.append(txt_row)
            cell_colors.append(clr_row)

        col_labels = [f"{label} {d}" for _, d, label in unpacked]

        fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(cols)), 0.6 * len(methods) + 1.5))
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            rowLabels=methods,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)

        # Bold the novel algorithm row label
        for (row, col), cell in table.get_celld().items():
            if col == -1 and row > 0 and "★" in methods[row - 1]:
                cell.set_text_props(fontweight="bold", color="#c0392b")
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#f0f0f0")

        fig.suptitle(
            "Quantitative Metrics Comparison",
            fontsize=13, fontweight="bold", y=0.95,
        )
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{output_prefix}_metrics_table.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Metrics table saved: {path}", flush=True)
        return path

    def _plot_metrics_bars(
        self,
        table_rows: List[Dict[str, Any]],
        metric_keys: List[tuple],
        algo_name: str,
        output_prefix: str = "novel",
        chart_title: str = "Metrics Comparison — Novel vs Baselines",
    ) -> str:
        """Grouped bar chart of key metrics across methods."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # metric_keys may be (key, dir) or (key, dir, label)
        def _unpack(t):
            return (t[0], t[1], t[2] if len(t) > 2 else t[0])

        methods = [r["method"] for r in table_rows]
        unpacked = [_unpack(t) for t in metric_keys]
        # Select subset of metrics that are naturally 0-1 for a clean chart
        bar_entries = [(k, label) for k, d, label in unpacked
                       if d == "↑" and k != "LOF_Median"]
        bar_keys = [k for k, _ in bar_entries]
        bar_labels = [label for _, label in bar_entries]
        n_methods = len(methods)
        n_metrics = len(bar_keys)
        x = np.arange(n_metrics)
        width = 0.8 / n_methods

        colors = ["#E74C3C", "#2E86AB", "#27AE60", "#F39C12", "#8E44AD"]

        fig, ax = plt.subplots(figsize=(max(8, 2 * n_metrics), 5))
        for i, row in enumerate(table_rows):
            vals = [
                float(row[k]) if row.get(k) is not None else 0
                for k in bar_keys
            ]
            offset = (i - n_methods / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=row["method"],
                          color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=10)
        ax.set_ylabel("Score (higher is better)", fontsize=10)
        ax.set_title(chart_title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{output_prefix}_metrics_bars.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Metrics bar chart saved: {path}", flush=True)
        return path

    # ------------------------------------------------------------------
    # Code execution helpers (adapted from standard Planner)
    # ------------------------------------------------------------------

    def _exec_custom_code(
        self,
        code_str: str,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        params: dict,
    ) -> tuple:
        """
        Execute an LLM-generated ``custom_embedding`` function.

        Returns (embedding, True, "") on success or (None, False, msg) on failure.
        Runs synchronously with no timeout — long-running algorithms are allowed
        to complete rather than be interrupted and forced to restart.
        """
        namespace: dict = {"np": np, "__builtins__": __builtins__}

        # Hard-block torch: this conda env has a numpy 2 / torch ABI mismatch
        # that produces "Numpy is not available" errors during forward passes.
        # Reject torch-using code before exec so Agent 10 sees a clear, fail-fast
        # message and the planner can fall back to PCA quickly instead of
        # burning the full recovery budget on impossible fixes.
        if _re.search(r'(^|\n)\s*(import\s+torch|from\s+torch)\b', code_str):
            return None, False, ("Runtime error: torch is unavailable in this "
                                 "environment (numpy 2/torch ABI mismatch). "
                                 "Rewrite custom_embedding using only numpy, "
                                 "scipy, and scikit-learn — do not import torch.")

        # Auto-correct common LLM mistake: n_iter → max_iter for sklearn TSNE
        if _re.search(r'TSNE\s*\([^)]*\bn_iter\b', code_str):
            logger.warning("Generated code uses 'n_iter' in TSNE — auto-correcting to 'max_iter'")
            code_str = _re.sub(r'\bn_iter\b(?=\s*=(?!=))', 'max_iter', code_str)

        try:
            exec(compile(code_str, "<novel_algorithm>", "exec"), namespace)
        except Exception as exc:
            return None, False, f"Compilation error: {exc}"

        fn = namespace.get("custom_embedding")
        if fn is None or not callable(fn):
            return None, False, "Code does not define a callable 'custom_embedding'"

        # Unsupervised mode: the designed algorithm must not see labels; metrics
        # scoring elsewhere still uses the true labels. This prevents a supervised
        # (label-using) embedding from gaining an unfair advantage.
        labels_for_fn = None if getattr(self, "_unsupervised", False) else labels
        try:
            with _time_limit(getattr(self, "_embedding_timeout", DEFAULT_EMBEDDING_TIMEOUT_SECONDS)):
                result = fn(data, labels_for_fn, params)
        except EmbeddingTimeout as exc:
            logger.warning("custom_embedding timed out: %s", exc)
            return None, False, (f"Timeout: {exc}. The algorithm is too slow on this "
                                 f"dataset (raise DR_EMBEDDING_TIMEOUT to allow more time).")
        except Exception as exc:
            tb = traceback.format_exc()
            return None, False, f"Runtime error: {exc}\n{tb}"

        if not isinstance(result, np.ndarray):
            return None, False, f"Return type is {type(result).__name__}, expected np.ndarray"
        if result.ndim != 2 or result.shape[1] != 2:
            return None, False, f"Return shape is {result.shape}, expected (n, 2)"
        if result.shape[0] != data.shape[0]:
            return None, False, f"Row count mismatch: returned {result.shape[0]} rows but data has {data.shape[0]}"
        if np.isnan(result).any() or np.isinf(result).any():
            return None, False, "Output contains NaN or Inf values"

        return result, True, ""

    def _test_custom_code(
        self, code_str: str, data: np.ndarray,
        labels: Optional[np.ndarray], params: dict,
    ) -> tuple:
        """Quick validation on a small subsample (max 500 rows)."""
        n = data.shape[0]
        if n > 500:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 500, replace=False)
            test_data, test_labels = data[idx], (labels[idx] if labels is not None else None)
        else:
            test_data, test_labels = data, labels
        _, ok, msg = self._exec_custom_code(code_str, test_data, test_labels, params)
        return ok, msg

    def _run_custom_code(
        self, code_str: str, data: np.ndarray,
        labels: Optional[np.ndarray], params: dict,
    ) -> tuple:
        """Run on full data, no timeout."""
        return self._exec_custom_code(code_str, data, labels, params)

    def _attempt_error_recovery(
        self,
        failing_code: str,
        error_message: str,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        current_params: dict,
        max_attempts: int = 4,
    ) -> tuple:
        """
        Use Agent 10 (ErrorRecovery) to diagnose and fix a runtime crash.

        Returns
        -------
        (fixed_code, fixed_params, True)  on success
        (original_code, current_params, False) if all attempts fail
        """
        n_samples, n_features = data.shape
        n_classes = int(np.unique(labels).size) if labels is not None else 0
        previous_attempts: list = []

        for attempt in range(1, max_attempts + 1):
            print(f"  🔧 Agent 10 recovery attempt {attempt}/{max_attempts} …", flush=True)
            ctx = {
                "failing_code": failing_code,
                "error_message": error_message,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "has_labels": labels is not None,
                "current_params": current_params,
                "previous_attempts": previous_attempts,
            }
            try:
                result = self.agent10.run(ctx)
            except Exception as exc:
                print(f"  ⚠️  Agent 10 LLM call failed: {exc}", flush=True)
                previous_attempts.append({"diagnosis": f"LLM error: {exc}", "error": str(exc)})
                continue

            diagnosis = result.get("diagnosis", "(none)")
            fix_summary = result.get("fix_summary", "(none)")
            fixed_code = result.get("fixed_code", "")
            fixed_params = result.get("fixed_params", current_params)
            confidence = result.get("confidence", 0)

            print(f"    Diagnosis : {diagnosis}", flush=True)
            print(f"    Fix       : {fix_summary}", flush=True)
            print(f"    Confidence: {confidence}", flush=True)

            if not fixed_code or not fixed_code.strip():
                print("  ⚠️  Agent 10 returned empty code.", flush=True)
                previous_attempts.append({"diagnosis": diagnosis, "error": "empty code returned"})
                continue

            # Validate the fix
            test_ok, test_msg = self._test_custom_code(fixed_code, data, labels, fixed_params)
            if test_ok:
                # Save recovery log
                log = {
                    "attempt": attempt,
                    "original_error": error_message[:500],
                    "diagnosis": diagnosis,
                    "fix_summary": fix_summary,
                    "confidence": confidence,
                }
                self._save_json(f"agent10_recovery_{attempt}.json", log)
                return fixed_code, fixed_params, True
            else:
                print(f"  ⚠️  Agent 10 fix still fails: {test_msg}", flush=True)
                previous_attempts.append({"diagnosis": diagnosis, "error": test_msg})
                # Feed the new error back for the next attempt
                error_message = test_msg
                failing_code = fixed_code

        return failing_code, current_params, False

    _CODE_RETRY_TEMPLATE = (
        "You previously designed this algorithm:\n"
        "{spec_json}\n\n"
        "Now produce ONLY the executable Python code that implements it.\n\n"
        "Function signature MUST be exactly:\n"
        "```\ndef custom_embedding(data, labels, params):\n```\n\n"
        "Rules:\n"
        "- data: numpy array (n_samples, n_features)\n"
        "- labels: numpy array (n_samples,) or None\n"
        "- params: dict of hyperparameters\n"
        "- Returns: numpy array (n_samples, 2)\n"
        "- ALL imports inside the function body\n"
        "- Must be self-contained — no external files\n"
        "- Must handle labels=None\n\n"
        "Return JSON with exactly these keys:\n"
        '{{\n'
        '  "custom_embedding_code": "the complete Python function as a string",\n'
        '  "default_params": {{"param_name": default_value}},\n'
        '  "param_descriptions": {{"param_name": "description"}}\n'
        '}}'
    )

    def _retry_code_generation(self, initial_spec: dict) -> tuple:
        """Focused second LLM call that asks only for the code."""
        spec_json = json.dumps(
            initial_spec.get("algorithm_specification", {}),
            indent=2, default=str,
        )
        prompt = self._CODE_RETRY_TEMPLATE.format(spec_json=spec_json)
        raw = self.llm.chat(
            system_message=(
                "You are an expert Python developer. Produce ONLY valid JSON "
                "containing executable code. No markdown, no commentary."
            ),
            user_message=prompt,
        )
        from .base_agent import BaseAgent
        parsed = BaseAgent._parse_json(raw)
        if parsed is None:
            logger.warning("Code retry: could not parse JSON")
            return "", {}
        code = parsed.get("custom_embedding_code", "")
        params = parsed.get("default_params", {})
        if code:
            logger.info("Code retry succeeded (%d chars)", len(code))
        return code, params

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _label_summary(
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]] = None,
    ) -> Optional[str]:
        """Generate label summary string."""
        if labels is None:
            return None
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        parts = []
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            name = label_names.get(int(u), str(u)) if label_names else str(u)
            parts.append(f"{name}: {c} ({100*c/total:.1f}%)")
        return "; ".join(parts)

    def _compute_patch_scores(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> tuple:
        """Cluster data into patches and compute reliability scores."""
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
                    emb = self.dr_engine.compute_embedding(
                        patch_data, method=method, n_pcs=min(20, patch_data.shape[1])
                    )
                    m = self.metrics_engine.compute_all(
                        hd_data=patch_data, ld_data=emb, labels=patch_lbl,
                        k=self._metrics_k, lof_neighbors=self._lof_neighbors
                    )

                    local_m = m.get("Neighborhood Preservation", {})
                    global_m = m.get("Global Structure & HD<->2D Correlation", {})

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
                patch_reliability_scores[pid] = float(np.mean(rel_vals))
            results.append(scores)

        # Map per-patch reliability to per-point arrays
        observed = list(patch_reliability_scores.values())
        fallback = float(np.median(observed)) if observed else 0.5
        point_reliability = np.array([
            patch_reliability_scores.get(int(pid), fallback)
            for pid in patch_labels
        ], dtype=float)

        q_low = float(np.percentile(point_reliability, 33))
        q_high = float(np.percentile(point_reliability, 67))

        if q_high - q_low < 0.02:
            med = float(np.median(point_reliability))
            q_low = med - 0.01
            q_high = med + 0.01

        point_difficulty = np.where(
            point_reliability >= q_high, "E",
            np.where(point_reliability <= q_low, "H", "M")
        )

        return results, point_difficulty, point_reliability

    def _compute_neighborhood_grid(
        self, data: np.ndarray, labels: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Grid-search over neighborhood sizes."""
        from sklearn.cluster import MiniBatchKMeans

        n = data.shape[0]
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
                    emb = self.dr_engine.compute_embedding(
                        patch, method="tsne", n_pcs=min(20, patch.shape[1])
                    )
                    m = self.metrics_engine.compute_all(
                        hd_data=patch, ld_data=emb, labels=patch_lbl
                    )
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
