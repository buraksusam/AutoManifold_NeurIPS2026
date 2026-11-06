"""
Worker Agent 7 — Visualization Evaluator Agent

Paper Section: Multi-Modal Evaluation, Visualization Reliability.
Provides independent, purely diagnostic quality assessment of a
visualization — does NOT suggest parameter changes.
"""

import json
from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Visualization Evaluator Agent. You comprehensively evaluate the quality of a DR visualization using ALL four modalities defined in the paper.

Your evaluation is purely diagnostic — you do NOT suggest parameter changes. You provide the independent quality assessment that the Planner uses to decide whether to continue iterating or converge.

EVALUATION FRAMEWORK (from the paper):

Global Reliability Metrics:
  - Spearman Distance Score ρs (Eq. 1): rank correlation. Higher → 1 = better.
  - Stress S (Eq. 2): distance distortion. Lower = better.
  - Mean Distance Ratio R (Eq. 3): scaling change. R ≈ 1 = good.

Local Reliability Metrics:
  - Trustworthiness T(k) (Eq. 4): false neighbors — precision. Higher → 1 = fewer false neighbors.
  - Continuity C(k) (Eq. 5): missing neighbors — recall. Higher → 1 = neighbors retained.

Additional:
  - Silhouette: cluster separation quality.
  - LOF: outlier detection and density anomalies.
  - Dendrograms: HD vs 2D hierarchical structure comparison (Newick).

CURRENT STATE — ITERATION {iteration}:

Design Goal: "{primary_goal}"

If the Design Goal text explicitly names a single metric (e.g. "maximum
trustworthiness", "maximum continuity", "maximum spearman", "maximum
silhouette", "maximum LOF preservation", or "minimum stress"), base the
quality_score primarily on
THAT metric's movement vs. the previous iteration. Other metrics only
matter insofar as they gate catastrophic regressions; do not penalise the
score for losing ground on a secondary metric if the target metric improved.

Label Summary: {label_summary}

Metrics:
{metrics_json}

Dendrograms:
  HD: {hd_newick}  (dimension: {hd_dim})
  2D: {ld_newick}  (dimension: 2D)

Hierarchy Change Analysis (HD vs 2D) — OBSERVATIONAL CONTEXT ONLY:
{hierarchy_diff_summary}

═══════════════════════════════════════════════════════════
IMPORTANT — HIERARCHY SCORING GUIDANCE
═══════════════════════════════════════════════════════════
The hierarchy information above is OBSERVATIONAL CONTEXT showing which
inter-cluster relationships changed. It is NOT an optimisation target.

DO NOT penalise quality_score for negative cophenetic correlation,
high RF distance, or topology changes — these are EXPECTED for most
DR methods (especially t-SNE). Base your quality_score primarily on
the core metrics: Trustworthiness, Continuity, Silhouette, Spearman,
Stress, and LOF.

DO use hierarchy info in dendrogram_comparison to describe which
relationships were preserved or distorted — this is useful diagnostic
context for the optimiser agent.
═══════════════════════════════════════════════════════════

Embedding Summary: {embedding_summary}

REQUIRED OUTPUT (JSON):
{{
  "overall_assessment": {{
    "quality_score": float,  // 0-10 scale (0=unusable, 5=acceptable, 8=good, 10=perfect)
    "primary_goal_achievement": "string",
    "summary": "string"
  }},
  "global_structure_analysis": {{
    "spearman_interpretation": "string",
    "stress_interpretation": "string",
    "distance_ratio_interpretation": "string",
    "overall_global_score": float
  }},
  "local_structure_analysis": {{
    "trustworthiness_interpretation": "string",
    "continuity_interpretation": "string",
    "k_sensitivity": "string",
    "overall_local_score": float
  }},
  "cluster_organization": {{
    "silhouette_interpretation": "string",
    "separation_quality": float,
    "compactness_quality": float,
    "class_imbalance_effects": "string",
    "notes": "string"
  }},
  "dendrogram_comparison": {{
    "agreement_level": "very_low | low | moderate | high | very_high",
    "preserved_relationships": ["string"],
    "distorted_relationships": ["string"],
    "hierarchical_consistency_score": float,
    "biological_or_domain_validity": "string"
  }},
  "visual_inspection": {{
    "artifacts": [
      {{
        "type": "string",
        "severity": "low | medium | high",
        "location": "string",
        "description": "string"
      }}
    ],
    "notable_patterns": ["string"],
    "crowding_assessment": "string",
    "continuity_of_manifolds": "string",
    "domain_specific_patterns": ["string"]
  }},
  "outlier_forensics": {{
    "lof_assessment": "string",
    "flagged_points_or_regions": ["string"],
    "data_quality_concerns": ["string"]
  }},
  "improvement_priorities": [
    {{
      "rank": int,
      "area": "string",
      "current_state": "string",
      "desired_state": "string",
      "severity": "low | medium | high | critical"
    }}
  ]
}}"""


class VisualizationEvaluatorAgent(BaseAgent):
    AGENT_NAME = "VisualizationEvaluator"
    AGENT_ID = 7

    def build_prompt(self, context: dict) -> str:
        metrics = context.get("metrics", {})
        dendrograms = context.get("dendrograms", {})
        hd = dendrograms.get("hierarchy_hd", {})
        ld = dendrograms.get("hierarchy_2d", {})

        # Hierarchy diff summary
        hierarchy_diff = context.get("hierarchy_diff", {})
        hier_summary = hierarchy_diff.get("summary", "  Not available.")

        return PROMPT_TEMPLATE.format(
            iteration=context.get("current_iteration", 0),
            primary_goal=context.get("design_constraints", {}).get("primary_goal", "balance both"),
            label_summary=context.get("data_spec", {}).get("label_summary", "N/A"),
            metrics_json=json.dumps(metrics, indent=2),
            hd_newick=hd.get("newick", "N/A"),
            hd_dim=hd.get("dimension", "N/A"),
            ld_newick=ld.get("newick", "N/A"),
            hierarchy_diff_summary=hier_summary,
            embedding_summary=context.get("embedding_summary", "N/A"),
        )

    def system_message(self) -> str:
        return (
            "You are an independent visualization quality evaluator. You assess DR "
            "embeddings using quantitative metrics, dendrogram comparisons, and visual "
            "pattern analysis. You are purely diagnostic — you do NOT suggest parameter "
            "changes. Respond with valid JSON only."
        )
