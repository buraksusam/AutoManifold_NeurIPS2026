"""
NoSkelAgentsPlanner — NoSkel-Agents: skeleton-free ("blank canvas") algorithm
design under the DR_Template_v2 constraints (docs/DR_Template_v2.md).

Subclasses AutoManifoldPlanner and overrides the design/synthesis/recovery phases.
Inherits all code-execution helpers (_exec_custom_code, _test_custom_code,
_run_custom_code, _attempt_error_recovery, _iterative_optimise, etc.).

Phase 1: Data Characterization    (inherited from AutoManifoldPlanner)
Phase 2: Open-Ended Design        (NoSkelDesignerAgent + DynamicAgents)
Phase 3: Code Synthesis           (Agent 8, noskel_agents prompt)
Phase 4: Implementation & Recovery (Agent 10 × 5 retries + t-SNE fallback)
Phase 5: Iterative Refinement     (inherited from AutoManifoldPlanner)
Phase 6: Visual Forensics         (inherited from AutoManifoldPlanner)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .config import OUTPUT_ROOT, PipelineConfig
from .planner_automanifold import AutoManifoldPlanner
from .agents.agent_noskel_designer import NoSkelDesignerAgent
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class NoSkelAgentsPlanner(AutoManifoldPlanner):
    """
    NoSkel-Agents: unconstrained, skeleton-free DR algorithm design pipeline.
    The designer has a blank canvas — no primitive vocabulary constraints.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.mode_name = "noskel_agents"
        self.template_name = "DR_Template_v2"
        self.designer_agent = NoSkelDesignerAgent(self.llm)
        # Agent 8 implements the designer's free-form spec (no primitive agents).
        from .agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
        self.agent8 = AlgorithmSynthesizerAgent(self.llm, mode="noskel_agents")

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
        """Execute the NoSkel-Agents algorithm design pipeline."""
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
        }

        self._stage("📊", "PHASE 1 — Data Characterization")
        self._phase1_characterize(data, labels)

        self._stage("🎨", "PHASE 2 — Open-Ended Algorithm Design")
        design_spec = self._phase2_design()

        self._stage("🏗️", "PHASE 3 — Code Synthesis")
        synth_result = self._phase3_synthesize_noskel(design_spec)

        self._stage("🔧", "PHASE 4 — Implementation & Recovery")
        impl_result = self._phase4_implement_and_recover(data, labels, label_names, synth_result)

        algo_name = impl_result.get("algo_name", "NoSkel-Agents Algorithm")
        self._novel_algo_name = algo_name

        # Phase 4b — Mathematical Representation & Pseudocode
        self._stage("📐", "PHASE 4b — Mathematical Representation & Pseudocode")
        algo_description = (
            (design_spec.get("description") or design_spec.get("brief_description") or "")
            if isinstance(design_spec, dict) else ""
        )
        math_spec = self._phase_math_pseudocode(
            algorithm_name=algo_name,
            code_str=impl_result.get("code", "") if not impl_result.get("used_fallback") else "",
            default_params=impl_result.get("params", {}),
            description=algo_description,
            output_prefix="noskel",
        )

        if impl_result.get("used_fallback"):
            embedding = impl_result["embedding"]
            # Store fallback embedding as best_result so Phase 6 works
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
        elif self.config.convergence.enable_iterative_optimization:
            self._stage("🔄", "PHASE 5 — Iterative Refinement")
            embedding = self._iterative_optimise(
                data, labels, label_names,
                impl_result["code"],
                impl_result["params"],
                algo_name,
            )
        else:
            self._stage("⏭", "PHASE 5 — Single-shot (iterative optimization disabled)")
            embedding = self._single_shot_embed(
                data, labels, label_names,
                impl_result["code"],
                impl_result["params"],
                algo_name,
            )

        self._stage("📈", "PHASE 6 — Baseline Comparison")
        best_metrics = self.best_result["metrics"] if self.best_result else {}
        comparison = self._compare_with_baselines(
            data,
            labels,
            label_names,
            embedding,
            best_metrics,
            algo_name,
            output_prefix="noskel",
            comparison_title="NoSkel-Agents Algorithm vs Baselines",
            metrics_plot_title="Metrics Comparison — NoSkel-Agents vs Baselines",
            metrics_json_label="NoSkel-Agents Algorithm",
        )
        self._save_json("noskel_comparison.json", comparison)

        self._stage("🔬", "PHASE 7 — Visual Forensics")
        vi_result = {}
        try:
            vi_result = self._phase7_visual_forensics(data, labels, label_names)
        except Exception as exc:
            logger.warning("Phase 6 (visual forensics) failed: %s", exc)

        final = {
            "mode": self.mode_name,
            "template_version": self.template_name,
            "design_spec": design_spec,
            "synthesis_result": synth_result,
            "implementation_result": impl_result,
            "math_pseudocode": math_spec,
            "best_result": self.best_result,
            "baseline_comparison": comparison,
            "optimization_history": self.optimization_history,
            "visual_forensics": vi_result,
        }
        report_path = os.path.join(output_dir, "final_noskel_report.json")
        with open(report_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final report saved → %s", report_path)
        return final

    # ------------------------------------------------------------------
    # Phase 2 — Open-Ended Design
    # ------------------------------------------------------------------

    def _phase2_design(self) -> Dict[str, Any]:
        """Multi-turn design loop using NoSkelDesignerAgent."""
        print("  [1/1] Running open-ended design loop …", flush=True)
        data_char = dict(self.context.get("agent1_output", {}))
        # Surface the optimization objective directly to the designer. Without
        # this the goal only reaches the designer indirectly (via Agent 1's
        # characterization), so distinct priorities barely diverge.
        goal = self.context.get("design_constraints", {}).get("primary_goal")
        if goal:
            data_char = {"design_objective": goal, **data_char}
        design_spec = self.designer_agent.run_design_loop(data_char)

        # Separate transcript from the spec before saving
        session_transcript = design_spec.pop("_session_transcript", [])
        forced = design_spec.pop("_forced_finalize", False)

        self._save_json("noskel_design_session.json", {
            "transcript": session_transcript,
            "forced_finalize": forced,
        })
        self._save_json("noskel_algorithm_spec.json", design_spec)

        algo_name = design_spec.get("algorithm_name", "?")
        print(f"  → Algorithm: {algo_name}", flush=True)
        if forced:
            print("  ⚠️  Design was force-finalized on last round", flush=True)

        self.context["design_spec"] = design_spec
        return design_spec

    # ------------------------------------------------------------------
    # Phase 3 — Code Synthesis
    # ------------------------------------------------------------------

    def _phase3_synthesize_noskel(self, design_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize code using Agent 8 in noskel_agents mode."""
        print("  [1/1] Synthesizing code (Agent 8 — noskel_agents mode) …", flush=True)
        ctx = {**self.context, "mode": self.mode_name, "design_spec": design_spec}
        result = self.agent8.run(ctx)
        self._save_json("agent8_final_synthesis.json", result)
        name = result.get("algorithm_specification", {}).get("name", "?")
        print(f"  → Synthesized: {name}", flush=True)
        return result

    # ------------------------------------------------------------------
    # Phase 4 — Implementation & Recovery (higher retry budget + t-SNE fallback)
    # ------------------------------------------------------------------

    def _phase4_implement_and_recover(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray],
        label_names: Optional[Dict[int, str]],
        synth_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate code, recover with Agent 10 (×5), fallback to t-SNE only."""
        code_str = synth_result.get("custom_embedding_code", "")
        default_params = synth_result.get("default_params", {})
        algo_name = synth_result.get("algorithm_specification", {}).get("name", "NoSkel-Agents Algorithm")
        result: Dict[str, Any] = {"algorithm_name": algo_name, "success": False, "used_fallback": False}

        # --- Code generation retry (up to 3×) if Agent 8 returned empty ---
        if not code_str:
            print("  ⚠️  No code from Agent 8 — retrying up to 3× …", flush=True)
            for attempt in range(1, 4):
                print(f"  → Code retry {attempt}/3 …", flush=True)
                code_str, default_params = self._retry_code_generation_noskel(attempt)
                if code_str:
                    print(f"  ✅ Code obtained on retry {attempt}", flush=True)
                    break

        if not code_str:
            print("  ❌ No code produced after 3 retries — falling back to t-SNE", flush=True)
            embedding = self._fallback_embedding(data)
            result.update({"used_fallback": True, "fallback_reason": "no code produced",
                           "embedding": embedding, "algo_name": algo_name})
            return result

        # Save generated code
        code_path = os.path.join(self.output_dir, "noskel_algorithm.py")
        with open(code_path, "w") as f:
            f.write(f"# {algo_name}\n# Generated by the NoSkel-Agents pipeline\n\n")
            f.write(code_str)
        print(f"  → Code saved: {code_path}", flush=True)

        # Validate on small sample
        print("  Validating code on small sample …", flush=True)
        test_ok, test_msg = self._test_custom_code(code_str, data, labels, default_params)

        if not test_ok:
            print(f"  ⚠️  Validation failed: {test_msg}", flush=True)
            print(f"  → Attempting recovery via Agent 10 (max {self.config.noskel_max_recovery_attempts} attempts) …", flush=True)
            fixed_code, fixed_params, recovery_ok = self._attempt_error_recovery(
                code_str, test_msg, data, labels, default_params,
                max_attempts=self.config.noskel_max_recovery_attempts,
            )
            if recovery_ok:
                code_str = fixed_code
                default_params = fixed_params
                rec_path = os.path.join(self.output_dir, "noskel_algorithm_recovered.py")
                with open(rec_path, "w") as f:
                    f.write(f"# Recovered from error: {test_msg[:200]}\n\n")
                    f.write(code_str)
                print(f"  ✅ Recovery succeeded → {rec_path}", flush=True)
                result["code_validation"] = {"ok": True, "recovered_by_agent10": True}
            else:
                print("  ❌ All recovery attempts failed — falling back to t-SNE", flush=True)
                embedding = self._fallback_embedding(data)
                result.update({"used_fallback": True, "fallback_reason": test_msg,
                               "embedding": embedding, "algo_name": algo_name,
                               "code_validation": {"ok": False, "message": test_msg}})
                return result
        else:
            print("  ✅ Code validated successfully", flush=True)
            result["code_validation"] = {"ok": True}

        result.update({"code": code_str, "params": default_params,
                       "algo_name": algo_name, "success": True})
        return result

    # ------------------------------------------------------------------
    # Helpers specific to NoSkelAgentsPlanner
    # ------------------------------------------------------------------

    def _fallback_embedding(self, data: np.ndarray) -> np.ndarray:
        """Use t-SNE as the sole fallback embedding for NoSkel-Agents mode."""
        print("  → Falling back to t-SNE …", flush=True)
        return self.dr_engine.compute_embedding(data, method="tsne")

    def _retry_code_generation_noskel(self, attempt: int) -> tuple:
        """Focused code-only LLM call using the design spec from context."""
        design_spec = self.context.get("design_spec", {})
        spec_json = json.dumps(design_spec, indent=2, default=str)[:3000]
        prompt = (
            f"Attempt {attempt}/3: The previous code generation produced no executable code.\n\n"
            f"Algorithm specification:\n{spec_json}\n\n"
            "Produce ONLY valid JSON with these keys:\n"
            "{\n"
            '  "custom_embedding_code": "COMPLETE Python function: def custom_embedding(data, labels, params): ...",\n'
            '  "default_params": {"param_name": default_value}\n'
            "}\n\n"
            "Rules:\n"
            "- Signature MUST be: def custom_embedding(data, labels, params):\n"
            "- Returns numpy array of shape (n_samples, 2)\n"
            "- ALL imports inside the function body\n"
            "- Handle labels=None\n"
            "- Use max_iter= not n_iter= for sklearn TSNE"
        )
        raw = self.llm.chat(
            system_message="You are an expert Python developer. Output only valid JSON with executable code.",
            user_message=prompt,
        )
        parsed = BaseAgent._parse_json(raw)
        if not parsed:
            return "", {}
        return parsed.get("custom_embedding_code", ""), parsed.get("default_params", {})
