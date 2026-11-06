"""
Worker Agent 5 — Outlier Identification Agent

DR_Template Section 1.4: Identify Impossible to Embed Points.
Identifies data points whose removal would significantly improve
embedding reliability, and recommends separate-visualization strategy.
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Outlier Identification Agent. Your task is to identify "impossible to embed" points — data points whose removal would significantly improve embedding reliability scores — following Section 1.4 of the DR_Template.

CONTEXT (from DR_Template Section 1.4):
"In general, there exist a set of high-dimensional data points, for which it is hard to preserve their distances with other data points. By removing these, the embedding reliability scores should significantly increase, which can be used as a metric for the AI agent to locate the points. Design an algorithm to identify these points and check its amount. It can be useful to remove these points from the plot, and visualise them separately."

INPUT:
- Data: n = {n_points}, D = {original_dimension}
- Baseline reliability metrics: {baseline_metrics}
- Difficulty classification: {difficulty_summary}
- LOF statistics: {lof_stats}

PER-POINT DIAGNOSTIC SCORES (pre-computed):
{per_point_scores}

TASKS:
1. Design an algorithm to identify impossible-to-embed points. The defining metric is: removal → measurable improvement in T(k), C(k), S.
2. Consider multiple indicators:
   - Points contributing most to Stress (high local distance distortion)
   - Points appearing as false neighbors (harming Trustworthiness)
   - Points losing their true neighbors (harming Continuity)
   - LOF-based density outliers
   - Points at manifold intersections or cluster boundaries
3. Quantify percentage of "impossible to embed" points.
4. Assess reliability improvement after removal.
5. Recommend separate visualization strategy (DR_Template Section 2.1 Strategy 4).

REQUIRED OUTPUT (JSON):
{{
  "identification_algorithm": {{
    "name": "string",
    "description": "string",
    "steps": [
      {{"step": int, "description": "string", "pseudocode": "string"}}
    ],
    "key_indicators_used": [
      {{
        "indicator": "string",
        "weight": float,
        "rationale": "string"
      }}
    ]
  }},
  "identified_points": {{
    "indices": [int],
    "count": int,
    "percentage_of_total": float,
    "characteristics": {{
      "common_features": ["string"],
      "spatial_distribution_in_HD": "string",
      "label_distribution": {{}},
      "potential_causes": ["string"]
    }}
  }},
  "impact_assessment": {{
    "before_removal": {{
      "trustworthiness": float,
      "continuity": float,
      "stress": float,
      "spearman": float,
      "silhouette": float
    }},
    "after_removal": {{
      "trustworthiness": float,
      "continuity": float,
      "stress": float,
      "spearman": float,
      "silhouette": float
    }},
    "improvement": {{
      "trustworthiness_delta": float,
      "continuity_delta": float,
      "stress_delta": float,
      "spearman_delta": float,
      "percentage_improvement_summary": "string"
    }}
  }},
  "visualization_strategy": {{
    "separate_plot_recommended": true,
    "plot_type": "string",
    "linking_method": "string",
    "treatment_in_main_plot": "string"
  }}
}}"""


class OutlierIdentificationAgent(BaseAgent):
    AGENT_NAME = "OutlierIdentification"
    AGENT_ID = 5

    def build_prompt(self, context: dict) -> str:
        data_spec = context["data_spec"]
        return PROMPT_TEMPLATE.format(
            n_points=data_spec["n_points"],
            original_dimension=data_spec["original_dimension"],
            baseline_metrics=context.get("baseline_metrics", "N/A"),
            difficulty_summary=context.get("agent2_output", {}).get("difficulty_classification", "N/A"),
            lof_stats=context.get("lof_stats", "N/A"),
            per_point_scores=context.get("per_point_scores", "Not yet computed — design the algorithm from the indicators above."),
        )

    def system_message(self) -> str:
        return (
            "You are a specialized agent that identifies data points that are impossible "
            "to embed reliably in 2D. You design outlier-detection algorithms grounded in "
            "DR reliability metrics. Respond with valid JSON only."
        )
