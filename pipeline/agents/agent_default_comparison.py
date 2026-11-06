"""
Worker Agent — Default-Params Comparison Agent

Compares the best result from iterative hyperparameter optimization with the
embedding produced by the same algorithm's *default* parameters. Used in
``optimization`` mode, where the pipeline is constrained to hyperparameter
tuning only (no algorithm design), to quantify what the optimizer achieved
relative to an out-of-the-box baseline.
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Default-Params Comparison Agent. Your job is to judge whether the hyperparameter optimization run produced a meaningfully better embedding than the same DR algorithm with its out-of-the-box default parameters.

ALGORITHM: {dr_method}

DEFAULT HYPERPARAMETERS (library defaults):
{default_params_json}

BEST HYPERPARAMETERS (from optimizer):
{best_params_json}

PARAMETERS THAT CHANGED (default → best):
{param_diff_json}

METRICS — DEFAULT RUN:
{default_metrics_json}

METRICS — BEST RUN:
{best_metrics_json}

METRIC DIRECTIONS:
  Higher is better: Trustworthiness, Continuity, Spearman, Silhouette
  Lower is better:  Stress, LOF_Median

TASK:
1. For each metric above, report the absolute delta (best − default) and a short verdict: "better", "worse", or "unchanged" (within ±0.005 is unchanged).
2. Weigh the deltas by the design goal: "{primary_goal}" — e.g., if the goal is local structure, trustworthiness/continuity dominate; if global, spearman/stress dominate; balanced means all matter.
3. Give an overall verdict: "optimization_helped" | "optimization_neutral" | "optimization_hurt".
4. Provide a 1–3 sentence rationale citing the most impactful metric shifts and the most influential parameter changes. Do not invent numbers — only cite the deltas you computed.
5. Recommend whether the user should adopt the optimized params (yes/no) and why.

REQUIRED OUTPUT (JSON):
{{
  "algorithm": "{dr_method}",
  "overall_verdict": "optimization_helped | optimization_neutral | optimization_hurt",
  "recommend_adopt_optimized_params": true,
  "rationale": "string — 1-3 sentences",
  "metric_comparison": [
    {{
      "metric": "string",
      "default_value": float_or_null,
      "best_value": float_or_null,
      "delta": float_or_null,
      "direction": "higher_better | lower_better",
      "verdict": "better | worse | unchanged"
    }}
  ],
  "influential_param_changes": [
    {{"parameter": "string", "from": value, "to": value, "impact": "string — short explanation"}}
  ],
  "caveats": ["string"]
}}
"""


class DefaultComparisonAgent(BaseAgent):
    AGENT_NAME = "DefaultComparison"
    AGENT_ID = 11

    def build_prompt(self, context: dict) -> str:
        import json

        return PROMPT_TEMPLATE.format(
            dr_method=context.get("dr_method", "?"),
            default_params_json=json.dumps(context.get("default_params", {}), indent=2, default=str),
            best_params_json=json.dumps(context.get("best_params", {}), indent=2, default=str),
            param_diff_json=json.dumps(context.get("param_diff", {}), indent=2, default=str),
            default_metrics_json=json.dumps(context.get("default_metrics_snapshot", {}), indent=2, default=str),
            best_metrics_json=json.dumps(context.get("best_metrics_snapshot", {}), indent=2, default=str),
            primary_goal=context.get("primary_goal", "balance both"),
        )

    def system_message(self) -> str:
        return (
            "You are a quantitative analyst comparing two DR embeddings of the "
            "same dataset produced by the same algorithm with different "
            "hyperparameters. Respond with valid JSON only."
        )
