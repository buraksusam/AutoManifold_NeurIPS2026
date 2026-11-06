"""
SingleLLMPlanner — SingleLLM: single-LLM blank-prompt baseline.

One LLM call, zero data context, zero planning
agents. The model is simply asked to design a novel dimensionality-reduction
algorithm and return runnable Python. We then take that code and hand it to our
existing evaluation machinery (metrics engine, baseline comparison, visual
forensics) so the output is directly comparable against AutoManifoldPlanner runs.

Phases:
  1. Blank-Prompt Design   — one LLM call, no data fed in
  2. Implementation        — validate + Agent 10 recovery (inherited)
  3. Single-shot Embed     — inherited from AutoManifoldPlanner
  4. Baseline Comparison   — inherited
  5. Visual Forensics      — inherited (Agent 9)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .base_agent import BaseAgent
from .config import OUTPUT_ROOT, PipelineConfig
from .planner_automanifold import AutoManifoldPlanner

logger = logging.getLogger(__name__)


BLANK_SYSTEM = (
    "You are a creative machine-learning researcher. Output only valid JSON "
    "containing a novel dimensionality reduction algorithm with executable Python code."
)


def _priority_clause(primary_goal: str) -> str:
    """Map the pipeline's primary_goal into a one-line priority directive."""
    g = (primary_goal or "").strip().lower()
    if not g or "balance" in g:
        return (
            "Priority: balance local neighborhood preservation with global "
            "structure (distance ranks / stress)."
        )
    if "local structure" in g or g == "local":
        return "Priority: prioritise LOCAL structure (neighborhood preservation)."
    if "global structure" in g or g == "global":
        return "Priority: prioritise GLOBAL structure (distance ranks, stress)."
    if "trustworthiness" in g:
        return (
            "Priority: maximise TRUSTWORTHINESS@k — the algorithm must avoid "
            "introducing false neighbors in 2D."
        )
    if "continuity" in g:
        return (
            "Priority: maximise CONTINUITY@k — the algorithm must avoid losing "
            "true high-dimensional neighbors in 2D."
        )
    if "spearman" in g:
        return (
            "Priority: maximise SPEARMAN rank correlation between HD and LD "
            "pairwise distances."
        )
    if "silhouette" in g:
        return (
            "Priority: maximise SILHOUETTE — cluster separation in 2D — without "
            "requiring labels at training time."
        )
    if "lof" in g:
        return (
            "Priority: minimise embedding outliers (LOF Outlier %) — avoid "
            "spurious isolated points in 2D."
        )
    if "stress" in g:
        return (
            "Priority: minimise STRESS — preserve absolute HD distances in 2D "
            "as faithfully as possible."
        )
    # Fallback: echo the raw goal so custom strings still propagate.
    return f"Priority: {primary_goal.strip()}."


def _build_blank_prompt(primary_goal: str) -> str:
    return (
        "Design a novel dimensionality reduction algorithm that projects data "
        "from high-dimensional space to 2D. Be creative — it does not need to "
        "resemble t-SNE, UMAP, or PacMAP, but it must actually work on real data.\n\n"
        f"{_priority_clause(primary_goal)} Your algorithm's core design, "
        "objective function, and default hyperparameters should reflect this "
        "priority.\n\n"
        "Return ONLY valid JSON with these keys:\n"
        "{\n"
        '  "algorithm_name": "short descriptive name",\n'
        '  "brief_description": "1-3 sentences explaining the core idea and how it serves the stated priority",\n'
        '  "custom_embedding_code": "COMPLETE Python function: def custom_embedding(data, labels, params): ...",\n'
        '  "default_params": {"param_name": default_value}\n'
        "}\n\n"
        "Hard requirements for custom_embedding_code:\n"
        "- Signature MUST be exactly: def custom_embedding(data, labels, params):\n"
        "- Return a numpy array of shape (n_samples, 2)\n"
        "- Put ALL imports inside the function body\n"
        "- Handle labels=None (do not assume supervision)\n"
        "- For sklearn TSNE use max_iter=, NOT n_iter=\n"
        "- Do not read files, do not hit the network, no plotting"
    )


class SingleLLMPlanner(AutoManifoldPlanner):
    """SingleLLM baseline; reuses AutoManifoldPlanner's execution + evaluation."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.mode_name = "singlellm"
        self.template_name = "SingleLLM_Blank_Prompt"

    # ------------------------------------------------------------------
    # Public entry point
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
            },
        }

        # Phase 1 — blank-prompt design
        self._stage("🎨", "PHASE 1 — Blank-Prompt Algorithm Design")
        design = self._phase1_blank_design()

        # Phase 2 — implementation + recovery
        self._stage("🔧", "PHASE 2 — Implementation & Recovery")
        impl = self._phase2_implement(data, labels, design)

        algo_name = impl.get("algo_name", "SingleLLM Algorithm")
        self._novel_algo_name = algo_name

        # Phase 2b — Mathematical Representation & Pseudocode
        self._stage("📐", "PHASE 2b — Mathematical Representation & Pseudocode")
        math_spec = self._phase_math_pseudocode(
            algorithm_name=algo_name,
            code_str=impl.get("code", "") if not impl.get("used_fallback") else "",
            default_params=impl.get("params", {}),
            description=design.get("brief_description", ""),
            output_prefix="singlellm",
        )

        # Phase 3 — single-shot embed (always single-shot for SingleLLM)
        if impl.get("used_fallback"):
            embedding = impl["embedding"]
            metrics = self.metrics_engine.compute_all(
                hd_data=data, ld_data=embedding, labels=labels,
                k=self._metrics_k, lof_neighbors=self._lof_neighbors,
            )
            self.best_result = {
                "iteration": 0,
                "embedding": embedding,
                "metrics": metrics,
                "hyperparameters": {},
                "dendrograms": {},
            }
        else:
            self._stage("⏭", "PHASE 3 — Single-shot Embedding")
            embedding = self._single_shot_embed(
                data, labels, label_names,
                impl["code"], impl["params"], algo_name,
            )

        # Phase 4 — baseline comparison
        self._stage("📈", "PHASE 4 — Baseline Comparison")
        best_metrics = self.best_result["metrics"] if self.best_result else {}
        comparison = self._compare_with_baselines(
            data, labels, label_names, embedding, best_metrics, algo_name,
            output_prefix="singlellm",
            comparison_title="SingleLLM Algorithm vs Baselines",
            metrics_plot_title="Metrics Comparison — SingleLLM vs Baselines",
            metrics_json_label="SingleLLM Algorithm",
        )
        self._save_json("singlellm_comparison.json", comparison)

        # Phase 5 — visual forensics
        self._stage("🔬", "PHASE 5 — Visual Forensics")
        vi_result: Dict[str, Any] = {}
        try:
            vi_result = self._phase7_visual_forensics(data, labels, label_names)
        except Exception as exc:
            logger.warning("SingleLLM visual forensics failed: %s", exc)

        final = {
            "mode": self.mode_name,
            "template_version": self.template_name,
            "design_spec": design,
            "implementation_result": impl,
            "math_pseudocode": math_spec,
            "best_result": self.best_result,
            "baseline_comparison": comparison,
            "visual_forensics": vi_result,
        }
        report_path = os.path.join(output_dir, "final_singlellm_report.json")
        with open(report_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final SingleLLM report saved → %s", report_path)
        return final

    # ------------------------------------------------------------------
    # Phase 1 — Blank-Prompt Design
    # ------------------------------------------------------------------

    def _phase1_blank_design(self) -> Dict[str, Any]:
        primary_goal = getattr(self.config, "primary_goal", "") or ""
        print(f"  [1/1] Blank-prompt single-LLM design (priority: {primary_goal!r}) …", flush=True)

        user_prompt = _build_blank_prompt(primary_goal)
        raw = self.llm.chat(
            system_message=BLANK_SYSTEM,
            user_message=user_prompt,
        )
        parsed = BaseAgent._parse_json(raw) or {}

        # Retry once if the model returned no code.
        if not parsed.get("custom_embedding_code"):
            print("  ⚠️  First call returned no code — retrying once", flush=True)
            raw = self.llm.chat(
                system_message=BLANK_SYSTEM,
                user_message=(
                    user_prompt
                    + "\n\nYour previous response did not include executable "
                      "custom_embedding_code. Try again and include it."
                ),
            )
            parsed = BaseAgent._parse_json(raw) or {}

        design = {
            "algorithm_name": parsed.get("algorithm_name", "SingleLLM Algorithm"),
            "brief_description": parsed.get("brief_description", ""),
            "custom_embedding_code": parsed.get("custom_embedding_code", ""),
            "default_params": parsed.get("default_params", {}) or {},
            "primary_goal": primary_goal,
            "prompt": user_prompt,
            "_raw_llm_output": raw[:8000] if isinstance(raw, str) else "",
        }
        self._save_json("singlellm_design.json", design)

        name = design["algorithm_name"]
        print(f"  → Algorithm: {name}", flush=True)
        if design["brief_description"]:
            print(f"     {design['brief_description']}", flush=True)
        self.context["design_spec"] = design
        return design

    # ------------------------------------------------------------------
    # Phase 2 — Implementation + Recovery
    # ------------------------------------------------------------------

    def _phase2_implement(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        design: Dict[str, Any],
    ) -> Dict[str, Any]:
        code_str = design.get("custom_embedding_code", "")
        default_params = design.get("default_params", {}) or {}
        algo_name = design.get("algorithm_name", "SingleLLM Algorithm")
        result: Dict[str, Any] = {
            "algorithm_name": algo_name,
            "success": False,
            "used_fallback": False,
        }

        if not code_str:
            print("  ❌ No code produced — falling back to t-SNE", flush=True)
            embedding = self._fallback_tsne(data)
            result.update({
                "used_fallback": True,
                "fallback_reason": "no code produced",
                "embedding": embedding,
                "algo_name": algo_name,
            })
            return result

        code_path = os.path.join(self.output_dir, "singlellm_algorithm.py")
        with open(code_path, "w") as f:
            f.write(f"# {algo_name}\n# Generated by the SingleLLM (blank-prompt) pipeline\n\n")
            f.write(code_str)
        print(f"  → Code saved: {code_path}", flush=True)

        print("  Validating code on small sample …", flush=True)
        test_ok, test_msg = self._test_custom_code(code_str, data, labels, default_params)

        if not test_ok:
            print(f"  ⚠️  Validation failed: {test_msg}", flush=True)
            print("  → Attempting recovery via Agent 10 …", flush=True)
            fixed_code, fixed_params, recovery_ok = self._attempt_error_recovery(
                code_str, test_msg, data, labels, default_params, max_attempts=4,
            )
            if recovery_ok:
                code_str = fixed_code
                default_params = fixed_params
                rec_path = os.path.join(self.output_dir, "singlellm_algorithm_recovered.py")
                with open(rec_path, "w") as f:
                    f.write(f"# Recovered from error: {test_msg[:200]}\n\n")
                    f.write(code_str)
                print(f"  ✅ Recovery succeeded → {rec_path}", flush=True)
                result["code_validation"] = {"ok": True, "recovered_by_agent10": True}
            else:
                print("  ❌ Recovery failed — falling back to t-SNE", flush=True)
                embedding = self._fallback_tsne(data)
                result.update({
                    "used_fallback": True,
                    "fallback_reason": test_msg,
                    "embedding": embedding,
                    "algo_name": algo_name,
                    "code_validation": {"ok": False, "message": test_msg},
                })
                return result
        else:
            print("  ✅ Code validated successfully", flush=True)
            result["code_validation"] = {"ok": True}

        result.update({
            "code": code_str,
            "params": default_params,
            "algo_name": algo_name,
            "success": True,
        })
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_tsne(self, data: np.ndarray) -> np.ndarray:
        print("  → Falling back to t-SNE …", flush=True)
        return self.dr_engine.compute_embedding(data, method="tsne")
