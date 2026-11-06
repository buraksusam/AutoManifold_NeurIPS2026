"""
Worker Agent 6 — Hyperparameter Optimization Agent

DR_Template Section 2.3 + Paper core methodology.
Implements the paper's iterative LLM-guided pipeline with dual
implicit/explicit scoring and chain-of-thought diagnostics.
"""

import json
from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Hyperparameter Optimization Agent. You implement Section 2.3 of the DR_Template by following the agentic AI pipeline from the paper "Explainable Iterative Data Visualisation Refinement via an LLM Agent."

METHODOLOGY (from the paper):
You act as an expert visual diagnostician. You receive multi-modal input and produce a structured JSON diagnostic report. You MUST populate the dendrogram_comparison and visual_inspection fields BEFORE generating recommendations — this enforces chain-of-thought reasoning where parameter suggestions are directly tied to resolving specific identified issues.

Apply BOTH scoring approaches:
  (a) Implicit scoring: Assign a holistic quality score (0–10) as a visual diagnostician, capturing artifacts that mathematical metrics miss.
  (b) Explicit scoring: Define weight vector W and compute composite objective O = Σ wi * Oi(Z(A,θ)) per Eq. 6 of the paper.

HYPERPARAMETER KNOWLEDGE (from the paper):
- t-SNE: perplexity (30→80), learning_rate (200→800), max_iter (1000→3000), n_pcs (20→50).
  Low perplexity (<5) fragments; high (>50) merges. Insufficient max_iter → stringy structures.
- UMAP: n_neighbors (local vs global), min_dist (dense packing vs broad shape), n_pcs.
- PaCMAP: n_neighbors, MN_ratio, FP_ratio.

CURRENT STATE — ITERATION {iteration}:

Algorithm: {dr_method}
Current Hyperparameters: {current_params}

Current Algorithm Code:
```python
{current_code}
```

Metrics:
{metrics_json}

Dendrograms:
  HD: {hd_newick}  (dimension: {hd_dim})
  2D: {ld_newick}  (dimension: 2D)

Hierarchy Change Analysis (HD vs 2D) — OBSERVATIONAL CONTEXT ONLY:
{hierarchy_diff_summary}

═══════════════════════════════════════════════════════════
CRITICAL — HIERARCHY SCORING GUIDANCE
═══════════════════════════════════════════════════════════
The hierarchy information above is INFORMATIONAL CONTEXT to help you
understand WHICH inter-cluster relationships changed during dimensionality
reduction. It is NOT an optimisation target.

DO NOT penalise the quality_score for:
  • Negative cophenetic correlation (expected for t-SNE, common for UMAP)
  • High Robinson-Foulds distance (expected for local methods)
  • Topology changes (all DR methods distort some global relationships)

DO use hierarchy info to:
  • Identify which cluster pairs swapped neighbours (diagnostic insight)
  • Suggest parameter tweaks that might improve BOTH local quality AND
    happen to reduce specific hierarchy distortions
  • Provide richer rationale in your assessment

Your quality_score MUST be based primarily on the core DR metrics:
Trustworthiness, Continuity, Silhouette, Spearman, Stress, LOF.
═══════════════════════════════════════════════════════════

Embedding Summary: {embedding_summary}
Label Summary: {label_summary}
Design Goal: "{primary_goal}"

If the Design Goal text explicitly names a single metric (e.g. "maximum
trustworthiness", "maximum continuity", "maximum spearman", "maximum
silhouette", "maximum LOF preservation", or "minimum stress"), that metric
is the SOLE optimization target for hyperparameter choices and code_updates. Every recommendation must
be justified by how it raises/lowers THAT metric; other metrics are secondary
and may be traded off to gain the target.

Previous Iteration History:
{history_summary}

═══════════════════════════════════════════════════════════
YOU CAN MODIFY BOTH PARAMETERS AND CODE
═══════════════════════════════════════════════════════════
1. "recommendations" — change hyperparameter values (as before)
2. "code_update" — if the algorithm logic itself needs fixing (e.g., better stitching, different pre-processing, refined fine-tuning), provide a COMPLETE replacement function.

If the code is working well and only parameters need adjusting, set "code_update" to null.
If you provide a code_update, it MUST use the same signature:
  def custom_embedding(data, labels, params):
and return an (n, 2) numpy array. All imports inside the function body.

⚠️  SKLEARN t-SNE PARAMETER WARNING — THIS IS A COMMON MISTAKE:
  The parameter for number of iterations in sklearn's TSNE is `max_iter`, NOT `n_iter`.
  `n_iter` was renamed to `max_iter` in sklearn >= 1.4 and will raise a TypeError.
  ALWAYS use: TSNE(max_iter=1000, ...)  — NEVER use: TSNE(n_iter=1000, ...)

═══════════════════════════════════════════════════════════
CRITICAL — AVOID STALE ITERATIONS
═══════════════════════════════════════════════════════════
Review the "Previous Iteration History" carefully. You MUST:
  • NOT repeat parameter combinations already tried (check the history).
  • If the history shows "⚠️ NO PARAMETER CHANGES" for a past iteration, your
    previous recommendations were NOT applied — adjust your strategy.
  • Each iteration MUST make meaningful changes. If small tweaks aren't working,
    try BOLD changes: large parameter shifts, different algorithmic approaches,
    or a code_update to change the algorithm logic itself.
  • If score is plateauing, do NOT suggest the same params again.
    Try a fundamentally different direction (e.g., much higher/lower perplexity,
    different n_neighbors range, add preprocessing, change the loss function).
  • "recommendations" must reference parameter names that actually exist in
    the current hyperparameters dict or in params used by the custom code.

REQUIRED OUTPUT (JSON — following paper's Listing 2):
{{
  "quality_score": float,
  "score_rationale": "string",
  "explicit_scoring": {{
    "metric_weights": {{
      "Trustworthiness": float,
      "Continuity": float,
      "Silhouette Score": float,
      "Spearman Correlation": float,
      "Stress-1": float,
      "LOF Median": float
    }},
    "weight_rationale": "string",
    "composite_score": float
  }},
  "overall_assessment": {{
    "key_strengths": ["string"],
    "key_weaknesses": ["string"],
    "metric_analysis": {{
      "local_preservation_summary": "string",
      "global_preservation_summary": "string",
      "cluster_separation_summary": "string"
    }}
  }},
  "dendrogram_comparison": {{
    "agreement_level": "very_low | low | moderate | high | very_high",
    "key_similarities": ["string"],
    "key_differences": ["string"],
    "topological_distortions": ["string"]
  }},
  "visual_inspection": {{
    "cluster_separation": "poor | moderate | good | excellent",
    "cluster_compactness": "poor | moderate | good | excellent",
    "artifacts": [
      {{
        "type": "string",
        "severity": "low | medium | high",
        "location": "string",
        "likely_cause": "string"
      }}
    ],
    "notable_patterns": ["string"],
    "domain_specific_observations": ["string"]
  }},
  "outlier_forensics": {{
    "lof_distribution_assessment": "string",
    "concerning_patterns": ["string"],
    "data_quality_flags": ["string"]
  }},
  "recommendations": [
    {{
      "parameter": "string",
      "current_value": "string",
      "suggested_value": "string",
      "rationale": "string",
      "expected_impact": "string",
      "priority": "high | medium | low"
    }}
  ],
  "code_update": "string or null — if provided, the COMPLETE replacement Python function",
  "code_update_rationale": "string or null — why the code was changed",
  "follow_up_metrics": ["string"]
}}"""


class HyperparameterOptAgent(BaseAgent):
    AGENT_NAME = "HyperparameterOpt"
    AGENT_ID = 6

    @staticmethod
    def _coerce_params(value) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            coerced = {}
            for item in value:
                if isinstance(item, dict):
                    key = item.get("parameter") or item.get("name")
                    if key is not None:
                        coerced[str(key)] = item.get("suggested_value", item.get("value"))
            return coerced
        return {}

    def build_prompt(self, context: dict) -> str:
        metrics = context.get("metrics", {})
        dendrograms = context.get("dendrograms", {})
        hd = dendrograms.get("hierarchy_hd", {})
        ld = dendrograms.get("hierarchy_2d", {})

        # Build rich history summary so the LLM knows what was already tried
        history = context.get("optimization_history", [])
        if history:
            lines = []
            prev_params = None
            for idx, h in enumerate(history):
                score = h.get("implicit_score", "?")
                # Score delta from previous iteration
                if idx > 0 and isinstance(score, (int, float)):
                    prev_score = history[idx - 1].get("implicit_score", 0)
                    if isinstance(prev_score, (int, float)):
                        delta = score - prev_score
                        delta_str = f" ({'+' if delta >= 0 else ''}{delta:.2f})"
                    else:
                        delta_str = ""
                else:
                    delta_str = " (baseline)" if isinstance(score, (int, float)) else ""

                lines.append(f"  ── Iteration {h['iteration']}: score={score}{delta_str} ──")

                # Parameters used
                params = self._coerce_params(h.get("hyperparameters", {}))
                lines.append(f"    Parameters: {json.dumps(params, default=str)}")

                # What actually changed from previous iteration
                p_delta = h.get("params_delta", {}) if isinstance(h.get("params_delta", {}), dict) else {}
                if prev_params is not None:
                    # Compute delta if not stored in record (backward compat)
                    if not p_delta:
                        p_delta = {}
                        all_keys = set(list(params.keys()) + list(prev_params.keys()))
                        for k in all_keys:
                            if params.get(k) != prev_params.get(k):
                                p_delta[k] = {"from": prev_params.get(k), "to": params.get(k)}
                    if p_delta:
                        changes = [f"{k}: {v['from']}→{v['to']}" for k, v in p_delta.items()]
                        lines.append(f"    What changed: {', '.join(changes)}")
                    else:
                        lines.append(f"    ⚠️ NO PARAMETER CHANGES from previous iteration — the same config was re-run")

                # Key weaknesses identified
                weaknesses = h.get("agent7_key_weaknesses", [])
                if weaknesses:
                    lines.append(f"    Weaknesses: {'; '.join(str(w) for w in weaknesses[:4])}")

                # Score rationale
                rationale = h.get("agent6_score_rationale", "")
                if rationale:
                    lines.append(f"    Rationale: {str(rationale)[:200]}")

                # All recommendations that were made
                all_recs = h.get("agent6_all_recommendations", h.get("agent6_key_recommendations", []))
                if all_recs:
                    rec_parts = []
                    for r in all_recs:
                        rec_parts.append(
                            f"{r.get('parameter','?')}: {r.get('current_value','?')}→"
                            f"{r.get('suggested_value','?')} [{r.get('priority','?')}]"
                        )
                    lines.append(f"    Recommendations made: {'; '.join(rec_parts)}")

                # Which recommendations were skipped (not applied)
                skipped = h.get("skipped_recommendations", [])
                if skipped:
                    skip_names = [r.get("parameter", "?") for r in skipped]
                    lines.append(f"    ⚠️ Skipped (failed to apply): {', '.join(skip_names)}")

                # Code update
                if h.get("code_updated"):
                    lines.append(f"    ✓ Code was updated this iteration")

                # Stale iteration warning
                if h.get("stale_iteration"):
                    lines.append(
                        f"    🚨 THIS WAS A STALE ITERATION — nothing changed. "
                        f"You MUST try a fundamentally different approach."
                    )

                lines.append("")  # blank line between iterations
                prev_params = params

            history_summary = "\n".join(lines)
        else:
            history_summary = "No prior iterations."

        # Hierarchy diff summary
        hierarchy_diff = context.get("hierarchy_diff", {})
        hier_summary = hierarchy_diff.get("summary", "  Not available.")

        # Collapse warning — populated by the planner when the previous iter's
        # embedding tripped the geometric / implausibility guardrails. We splice
        # it directly into the history summary so the model sees it next to the
        # iteration that produced it.
        collapse_warning = context.get("collapse_warning")
        if collapse_warning and collapse_warning.get("tripped"):
            reasons = collapse_warning.get("reasons") or []
            cw_lines = [
                "",
                "═══════════════════════════════════════════════════════════",
                "🛑 COLLAPSE GUARDRAIL TRIPPED ON THE CURRENT ITERATION",
                "═══════════════════════════════════════════════════════════",
                "The embedding produced this iteration is geometrically "
                "degenerate (mass pile-up, exploded distance ratios, or a "
                "saturated single metric while neighborhood metrics regress).",
                "Reasons reported by the planner:",
            ]
            for r in reasons:
                cw_lines.append(f"  • {r}")
            cw_lines.extend([
                "",
                "MANDATORY behavior for your recommendations this turn:",
                "  • Do NOT keep amplifying the parameter that was driving the "
                "primary metric upward (e.g. silhouette_weight, learning_rate, "
                "max_iter, exaggeration). Doubling it again will not help — "
                "it is what produced the collapse.",
                "  • Re-introduce the counter-balancing force the optimizer "
                "lost: heavier KL / repulsive term, gradient-norm cap on the "
                "regularizer, smaller learning rate, fewer iterations, OR a "
                "code_update that bounds the offending gradient.",
                "  • Treat the previous best non-collapsed iteration as the "
                "anchor — propose a small perturbation around it, NOT another "
                "step in the direction that just collapsed.",
                "  • Your quality_score for THIS iteration MUST reflect the "
                "collapse; it cannot be ≥ 7 when the guardrail tripped.",
                "═══════════════════════════════════════════════════════════",
                "",
            ])
            history_summary = history_summary + "\n" + "\n".join(cw_lines)

        return PROMPT_TEMPLATE.format(
            iteration=context.get("current_iteration", 0),
            dr_method=context.get("dr_method", "umap"),
            current_params=json.dumps(context.get("current_hyperparameters", {}), indent=2),
            current_code=context.get("current_algorithm_code", "# No custom code — using default DR engine"),
            metrics_json=json.dumps(metrics, indent=2),
            hd_newick=hd.get("newick", "N/A"),
            hd_dim=hd.get("dimension", "N/A"),
            ld_newick=ld.get("newick", "N/A"),
            hierarchy_diff_summary=hier_summary,
            embedding_summary=context.get("embedding_summary", "N/A"),
            label_summary=context.get("data_spec", {}).get("label_summary", "N/A"),
            primary_goal=context.get("design_constraints", {}).get("primary_goal", "balance both"),
            history_summary=history_summary,
        )

    def system_message(self) -> str:
        return (
            "You are an expert visual diagnostician for dimensionality reduction. "
            "You analyze multi-modal inputs (metrics, dendrograms, plots) and produce "
            "structured JSON diagnostics following the methodology of the paper "
            "'Explainable Iterative Data Visualisation Refinement via an LLM Agent'. "
            "You MUST fill dendrogram_comparison and visual_inspection BEFORE recommendations. "
            "Respond with valid JSON only."
        )
