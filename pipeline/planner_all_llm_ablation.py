"""AllLLMAblationPlanner — stripped legacy all_llm mode.

The legacy all_llm ("blank canvas") designer spawns its OWN dynamic base-agent
workers (DynamicAgent), not the framework's defined agents 1-6. This ablation
keeps only that planner + its spawned workers -> synthesis -> single-shot
embedding, and removes:

  * Agent 1 data characterization (no upstream _phase1_characterize),
  * baseline scoring / calibration (legacy all_llm already omits it),
  * hyperparameter / iterative optimization (single-shot only, no Agent 6/7),
  * all post-hoc agents (Agent 11 math+pseudocode, baseline comparison,
    Agent 9 visual forensics, Agent 12 novelty validation).

The Agent 10 error-recovery step is retained: it is part of the code-execution
correctness pipeline (needed to obtain a runnable embedding), not a post-hoc
reporting agent.

Runs unsupervised when planner._unsupervised is True (labels withheld from the
designed algorithm, used only for scoring), consistent with the other
unsupervised runs.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .planner_all_llm import AllLLMPlanner
from .config import OUTPUT_ROOT

logger = logging.getLogger(__name__)


class AllLLMAblationPlanner(AllLLMPlanner):
    def __init__(self, config):
        super().__init__(config)
        self.mode_name = "all_llm_ablation"
        # No iterative hyperparameter optimization — force single-shot.
        self.config.convergence.enable_iterative_optimization = False

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
        }
        # NOTE: no _phase1_characterize (Agent 1 removed) and no baseline
        # calibration -> context has no agent1_output / baseline_profile.

        self._stage("🎨", "PHASE — Open-Ended Design (planner + spawned workers)")
        design_spec = self._phase2_design()

        self._stage("🏗️", "PHASE — Code Synthesis")
        synth_result = self._phase3_synthesize_all_llm(design_spec)

        self._stage("🔧", "PHASE — Implementation & Recovery")
        impl_result = self._phase4_implement_and_recover(data, labels, label_names, synth_result)

        algo_name = impl_result.get("algo_name", "AllLLM-Ablation Algorithm")
        self._novel_algo_name = algo_name

        if impl_result.get("used_fallback"):
            embedding = impl_result["embedding"]
        else:
            self._stage("⏭", "PHASE — Single-shot embedding (no hyperparameter optimization)")
            embedding = self._single_shot_embed(
                data, labels, label_names,
                impl_result["code"], impl_result["params"], algo_name,
            )

        metrics = self.metrics_engine.compute_all(
            hd_data=data, ld_data=embedding, labels=labels,
            k=self._metrics_k, lof_neighbors=self._lof_neighbors,
        )
        self.best_result = {"iteration": 0, "embedding": embedding, "metrics": metrics}
        np.save(os.path.join(output_dir, "best_embedding.npy"), embedding)

        final = {
            "mode": self.mode_name,
            "template_version": self.template_name,
            "algorithm_name": algo_name,
            "design_spec": design_spec,
            "used_fallback": bool(impl_result.get("used_fallback")),
            "removed_components": [
                "agent1_data_characterization", "baseline_scoring",
                "hyperparameter_optimization",
                "post_hoc:math_pseudocode", "post_hoc:baseline_comparison",
                "post_hoc:visual_forensics", "post_hoc:novelty_validation",
            ],
        }
        with open(os.path.join(output_dir, "final_all_llm_ablation_report.json"), "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final report saved → %s/final_all_llm_ablation_report.json", output_dir)
        return final
