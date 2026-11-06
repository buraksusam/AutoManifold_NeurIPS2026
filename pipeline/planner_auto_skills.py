"""AutoSkillsPlanner — Auto-Skills: autonomous LLM agent loop over skills.

Exposes both skills (each ``pipeline/skills/auto_skills/*/SKILL.md`` becomes
one tool, orchestrated by ``PLANNER.md``) and deterministic compute
helpers (baselines, PCA, neighborhood grid, code execution, distortion
metrics) AS TOOLS. The LLM autonomously decides which tool to call, in what
order, based on the planner prompt's workflow philosophy.

Termination is signalled by the model calling the ``finalize_design`` tool.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .agent_loop import AgentLoop, SessionState
from .config import OUTPUT_ROOT, PipelineConfig
from .engines.clustering_engine import ClusteringEngine
from .engines.dr_engine import DRVisualizationEngine
from .engines.metrics_engine import QuantitativeAnalystEngine
from .llm_client import LLMClient
from .tools import (
    ToolRegistry,
    register_compute_tools,
    register_finalize_tool,
    register_skill_tools,
)
from .visualizations import (
    plot_class_distortion_summary,
    plot_dendrogram_comparison,
    plot_distortion_heatmap,
    plot_hierarchy_change_detail,
    plot_highlighted_areas,
)

logger = logging.getLogger(__name__)

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills", "auto_skills")
PLANNER_PROMPT_PATH = os.path.join(SKILLS_DIR, "PLANNER.md")


class AutoSkillsPlanner:
    """Auto-Skills — autonomous LLM agent loop with tool-calling.

    Stands alone from :class:`AutoManifoldPlanner`: its deterministic helpers
    are exposed as compute tools and dispatched via the agent loop.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        self.mode_name = "auto_skills"
        self.template_name = "DR_Template_v2"

        # Auto-Skills defaults to high-effort thinking when not otherwise set.
        if not getattr(self.config.llm, "reasoning_effort", None):
            self.config.llm.reasoning_effort = "high"

        # Engines reused across compute tools
        self.dr_engine = DRVisualizationEngine()
        self.metrics_engine = QuantitativeAnalystEngine()
        self.clustering_engine = ClusteringEngine()

        # Hooks set by app.py / the front-end wiring layer
        self.skill_log_callback = None
        self.skill_stream_callback = None
        self.skill_thinking_callback = None
        self.tool_call_callback = None

        # State exposed for tests / inspectors
        self.context: Dict[str, Any] = {}
        self.output_dir: str = OUTPUT_ROOT

    # ------------------------------------------------------------------
    # Public entrypoint
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
            "data_spec": {
                "n_points": int(n),
                "original_dimension": int(d),
                "target_dimension": 2,
                "label_summary": self._label_summary(labels, label_names),
            },
        }

        state = SessionState(
            data=data,
            labels=labels,
            label_names=label_names,
            primary_goal=self.config.primary_goal,
            output_dir=output_dir,
            dr_engine=self.dr_engine,
            metrics_engine=self.metrics_engine,
            clustering_engine=self.clustering_engine,
            metrics_k=self.config.metrics.neighborhood_k,
            lof_neighbors=self.config.metrics.lof_neighbors,
        )

        registry = ToolRegistry()
        register_compute_tools(registry)
        skill_names = register_skill_tools(registry, skills_dir=SKILLS_DIR)
        register_finalize_tool(registry)

        planner_prompt = Path(PLANNER_PROMPT_PATH).read_text()

        loop = AgentLoop(
            llm_client=self.llm,
            planner_prompt=planner_prompt,
            tool_registry=registry,
            session_state=state,
            log_callback=self.skill_log_callback,
            stream_callback=self.skill_stream_callback,
            thinking_callback=self.skill_thinking_callback,
            tool_call_callback=self.tool_call_callback,
            skill_names=skill_names,
        )

        loop.add_initial_user_message(self._build_initial_message(n, d, labels, label_names))

        final_report = loop.run()
        final_report = dict(final_report) if isinstance(final_report, dict) else {"raw": final_report}
        final_report.setdefault("mode", self.mode_name)
        final_report.setdefault("template_version", self.template_name)
        final_report["iteration_count"] = loop.iteration_count
        final_report["code_attempts"] = [
            {k: v for k, v in atp.items() if k != "code"}  # drop heavy code blob
            for atp in state.code_attempts
        ]

        # Post-finalize: render the same plots + metrics tables AutoManifoldPlanner
        # produces. Without this, the agent loop ends with only a JSON report
        # — no visual artifacts. Wrapped in try/except so a plotting failure
        # never loses the JSON report.
        try:
            viz_artifacts = self._generate_final_visualizations(
                state=state,
                final_report=final_report,
                data=data,
                labels=labels,
                label_names=label_names,
                output_dir=output_dir,
            )
            if viz_artifacts:
                final_report["visualizations"] = viz_artifacts
        except Exception as exc:
            logger.exception("Final visualization phase failed: %s", exc)
            final_report.setdefault("visualizations", {})["error"] = str(exc)

        report_path = os.path.join(output_dir, "final_auto_skills_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        logger.info("Final report saved → %s", report_path)
        return final_report

    # ------------------------------------------------------------------
    # Final visualization phase
    # ------------------------------------------------------------------

    def _generate_final_visualizations(
        self,
        state: SessionState,
        final_report: Dict[str, Any],
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Render the same artifacts AutoManifoldPlanner produces post-finalize:
        final scatter, distortion heatmap, per-class summary, dendrogram
        comparison, hierarchy change detail, optional highlighted areas, and
        the metrics comparison table + bars vs baselines.

        Each step is independently try/except'd so one failure doesn't kill
        the rest.
        """
        artifacts: Dict[str, Any] = {}

        handle, embedding = self._resolve_final_embedding(state, final_report)
        if embedding is None:
            logger.warning("[finalize-viz] no final embedding found — skipping plots.")
            return {"warning": "no final embedding to visualize"}
        artifacts["embedding_handle"] = handle

        algo_name = (
            final_report.get("algorithm_name")
            or final_report.get("name")
            or "Novel Algorithm"
        )

        logger.info("[finalize-viz] rendering plots for handle=%s, algo=%r", handle, algo_name)

        # 1. Final scatter via dr_engine
        try:
            scatter_path = os.path.join(output_dir, f"final_{handle}.png")
            self.dr_engine.render_plot(
                embedding=embedding,
                labels=labels,
                label_names=label_names,
                title=f"★ {algo_name}",
                save_path=scatter_path,
            )
            artifacts["final_scatter"] = scatter_path
        except Exception as exc:
            logger.warning("[finalize-viz] final scatter failed: %s", exc)

        # 2. Pointwise metrics on full data
        try:
            pw = self.metrics_engine.compute_pointwise_metrics(
                hd_data=data, ld_data=embedding,
                k=self.config.metrics.neighborhood_k,
            )
            composite = pw["composite_distortion"]
            pw_trust = pw["pointwise_trust"]
            pw_cont = pw["pointwise_cont"]
            pw_stress = pw.get("pointwise_stress")
            if pw_stress is None:
                pw_stress = np.zeros_like(composite)
        except Exception as exc:
            logger.warning("[finalize-viz] pointwise metrics failed: %s", exc)
            return artifacts

        # 3. Distortion heatmap
        try:
            artifacts["distortion_heatmap"] = plot_distortion_heatmap(
                embedding=embedding,
                composite_distortion=composite,
                pointwise_trust=pw_trust,
                pointwise_cont=pw_cont,
                pointwise_stress=pw_stress,
                labels=labels,
                label_names=label_names,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.warning("[finalize-viz] distortion heatmap failed: %s", exc)

        # 4. Per-class distortion summary
        if labels is not None:
            try:
                artifacts["class_distortion_summary"] = plot_class_distortion_summary(
                    embedding=embedding,
                    composite_distortion=composite,
                    pointwise_trust=pw_trust,
                    pointwise_cont=pw_cont,
                    labels=labels,
                    label_names=label_names,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.warning("[finalize-viz] per-class summary failed: %s", exc)

        # 5. Dendrograms + hierarchy diff
        dendrograms = None
        hierarchy_diff = None
        try:
            dendrograms = self.clustering_engine.compute_dendrograms(
                hd_data=data, ld_data=embedding, labels=labels,
            )
            hierarchy_diff = self.clustering_engine.compute_hierarchy_diff(
                dendrograms, label_names=label_names, dr_method="custom",
            )
        except Exception as exc:
            logger.warning("[finalize-viz] dendrograms/hierarchy diff failed: %s", exc)

        if dendrograms:
            try:
                artifacts["dendrogram_comparison"] = plot_dendrogram_comparison(
                    dendrograms=dendrograms,
                    hierarchy_diff=hierarchy_diff,
                    label_names=label_names,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.warning("[finalize-viz] dendrogram comparison failed: %s", exc)
            if hierarchy_diff:
                try:
                    artifacts["hierarchy_change_detail"] = plot_hierarchy_change_detail(
                        dendrograms=dendrograms,
                        hierarchy_diff=hierarchy_diff,
                        embedding=embedding,
                        labels=labels,
                        label_names=label_names,
                        output_dir=output_dir,
                    )
                except Exception as exc:
                    logger.warning("[finalize-viz] hierarchy change detail failed: %s", exc)

        # 6. Highlighted areas if interpret-forensics produced regions.
        regions = (
            final_report.get("highlighted_regions")
            or (final_report.get("interpret_forensics") or {}).get("highlighted_regions")
            or (final_report.get("forensic_interpretation") or {}).get("highlighted_regions")
        )
        if regions:
            try:
                artifacts["highlighted_areas"] = plot_highlighted_areas(
                    embedding=embedding,
                    labels=labels,
                    label_names=label_names,
                    highlighted_regions=regions,
                    composite_distortion=composite,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.warning("[finalize-viz] highlighted areas failed: %s", exc)

        # 7. Metrics comparison vs baselines (table + bars + side-by-side scatter).
        try:
            from .planner_automanifold import AutoManifoldPlanner

            delegate = AutoManifoldPlanner(self.config)
            delegate.output_dir = output_dir
            novel_metrics = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self.config.metrics.neighborhood_k,
                lof_neighbors=self.config.metrics.lof_neighbors,
            )
            comparison = delegate._compare_with_baselines(
                data=data,
                labels=labels,
                label_names=label_names,
                novel_embedding=embedding,
                novel_metrics=novel_metrics,
                algo_name=algo_name,
                output_prefix="auto_skills",
                comparison_title=f"{algo_name} vs Baselines",
                metrics_plot_title="Metrics Comparison — Auto-Skills vs Baselines",
                metrics_json_label=algo_name,
            )
            artifacts["metrics_comparison"] = {
                "comparison_plot": comparison.get("comparison_plot"),
                "comparison_table_rows": comparison.get("comparison_table"),
            }
        except Exception as exc:
            logger.warning("[finalize-viz] metrics comparison failed: %s", exc)

        return artifacts

    @staticmethod
    def _resolve_final_embedding(
        state: SessionState,
        final_report: Dict[str, Any],
    ):
        """Pick the best-guess final embedding handle:
        1. final_report["embedding_handle"] if present and valid;
        2. last successful entry in state.code_attempts with a stored handle;
        3. highest-numbered novel_attempt_* in state.embeddings;
        4. None.
        """
        candidate = final_report.get("embedding_handle")
        if isinstance(candidate, str) and candidate in state.embeddings:
            return candidate, state.embeddings[candidate]

        for atp in reversed(state.code_attempts):
            if atp.get("ok") and atp.get("handle") in state.embeddings:
                h = atp["handle"]
                return h, state.embeddings[h]

        novel_handles = [h for h in state.embeddings if h.startswith("novel_attempt_")]
        if novel_handles:
            def _idx(h: str) -> int:
                try:
                    return int(h.rsplit("_", 1)[-1])
                except Exception:
                    return -1
            h = max(novel_handles, key=_idx)
            return h, state.embeddings[h]

        return None, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_message(
        self,
        n: int,
        d: int,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
    ) -> str:
        primary = self.config.primary_goal
        label_summary = self._label_summary(labels, label_names) or "no labels available"
        lines = [
            "# Initial design context",
            "",
            "## Dataset",
            f"- n_points: {n}",
            f"- original_dimension: {d}",
            "- target_dimension: 2",
            f"- label_summary: {label_summary}",
            "",
            "## Design goal",
            f"- primary_goal: {primary}",
            "- novelty_target: compositional_novelty",
            "- computational_budget: moderate",
            "",
            "## Available primitives",
            "- affinity: euclidean_knn, mutual_knn, snn, geodesic, lof_weighted, "
            "learnable, perplexity_gaussian, umap_fuzzy",
            "- cost: kl_divergence, cross_entropy, spectral, force_directed, "
            "mutual_information",
            "- ld_kernel: student_t, gaussian_ld, umap_ab",
            "- optimization: sgd, adam, spectral_eigen, fft_accelerated",
            "",
            "Begin by characterizing the data and running baselines so the rest of "
            "the design has grounded inputs. Use the available skills + compute "
            "tools and call `finalize_design` when complete.",
        ]
        return "\n".join(lines)

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
            parts.append(f"{name}: {c} ({100 * c / total:.1f}%)")
        return "; ".join(parts)
