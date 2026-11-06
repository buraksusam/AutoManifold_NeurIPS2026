"""
Worker Agent 3 — Maximal Embeddable Neighborhood Analyzer Agent

DR_Template Section 1.2: Identify Maximal Embeddable Neighborhoods — see docs/DR_Template.md.
Estimates the largest local neighbourhood radius that can be embedded
with acceptable reliability via grid search or predictive modelling.
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Maximal Embeddable Neighborhood Analyzer Agent. Your task is to estimate, for a set of target points, the largest local neighborhood radius that can be embedded with acceptable reliability — following Section 1.2 of the DR_Template.

CONTEXT (from DR_Template Section 1.2):
"Estimate the radius of a local neighborhood that can be embedded approximately isometrically, for a set of target points."

IMPORTANT — TWO DIFFERENT k's:
  (a) `optimal_k` / maximal embeddable neighborhood
      = UPPER BOUND on patch size for Strategy-2 patch stitching.
      Typically hundreds on clean data. DO NOT use as perplexity.
  (b) `recommended_affinity_k` / perplexity for HD-affinity construction
      = the typical neighbourhood size used by t-SNE / UMAP for LOCAL structure.
      Always in [10, 50] for "local" primary_goal; up to ~100 only for "global".
      Using (a) as (b) destroys local structure — this is a common failure mode.

INPUT:
- Data: n = {n_points}, D = {original_dimension}
- Design goal: "{primary_goal}"
- Difficulty classification from Agent 2: {difficulty_summary}
- Acceptable reliability threshold: T(k) ≥ {T_min}, C(k) ≥ {C_min}, S ≤ {S_max}

GRID SEARCH RESULTS (pre-computed):
{grid_search_results}

PROCEDURE (from DR_Template):
Approach 1 (Grid Search):
  1. For target points, define neighborhoods by increasing radius k.
  2. Embed each neighborhood with t-SNE, compute T(k), C(k), S.
  3. Find the LARGEST k where scores stay above threshold.

Approach 2 (Predictive Model):
  1. Extract geometric features (curvature, density, intrinsic dim).
  2. Build regression model to predict maximal k from features.

The grid search scores above have been pre-computed. Analyze them to determine:
  1. The optimal maximal neighbourhood radius globally and per difficulty group.
  2. How this varies across high-curvature, sparse, dense, and boundary regions.
  3. Patch sizing and overlap recommendations for the stitching phase.
  4. SEPARATELY, the recommended affinity-construction k / perplexity for
     downstream t-SNE / UMAP recipes, keyed to the stated primary_goal.
     - "local"    → recommended_affinity_k ∈ [10, 50]   (typical: 30)
     - "balanced" → recommended_affinity_k ∈ [15, 75]   (typical: 30–50)
     - "global"   → recommended_affinity_k ∈ [30, 150]  (typical: 50–100)

REQUIRED OUTPUT (JSON):
{{
  "methodology": {{
    "approach_used": "Approach 1 | Approach 2 | Hybrid",
    "n_target_points_sampled": int,
    "grid_search_details": {{
      "k_values_tested": [int],
      "reliability_threshold_used": {{"T_min": float, "C_min": float, "S_max": float}}
    }},
    "predictive_model_details": {{
      "features_used": ["string"],
      "model_type": "string",
      "model_accuracy": float
    }}
  }},
  "maximal_neighborhoods": {{
    "global_estimate": {{
      "optimal_k": int,
      "optimal_radius_distance": float,
      "confidence": float
    }},
    "per_difficulty_group": {{
      "H_patches": {{"avg_max_k": int, "range": [int, int]}},
      "M_patches": {{"avg_max_k": int, "range": [int, int]}},
      "E_patches": {{"avg_max_k": int, "range": [int, int]}}
    }},
    "local_variations": {{
      "high_curvature_regions": {{"max_k": int, "notes": "string"}},
      "sparse_regions": {{"max_k": int, "notes": "string"}},
      "dense_regions": {{"max_k": int, "notes": "string"}},
      "cluster_boundary_regions": {{"max_k": int, "notes": "string"}}
    }}
  }},
  "stitching_implications": {{
    "recommended_patch_size": int,
    "expected_overlap_needed": "string",
    "patch_boundary_strategy": "string"
  }},
  "affinity_construction_guidance": {{
    "recommended_affinity_k": int,
    "recommended_perplexity_range": [int, int],
    "justification": "string (must reference primary_goal and cite that this is NOT the maximal k)",
    "do_not_confuse_with": "optimal_k from maximal_neighborhoods (that is patch-sizing only)"
  }}
}}"""


class NeighborhoodAnalyzerAgent(BaseAgent):
    AGENT_NAME = "NeighborhoodAnalyzer"
    AGENT_ID = 3

    def build_prompt(self, context: dict) -> str:
        data_spec = context["data_spec"]
        patch_cfg = context.get("patch_config", {})
        design_constraints = context.get("design_constraints", {})
        primary_goal = design_constraints.get("primary_goal", "balance both")
        return PROMPT_TEMPLATE.format(
            n_points=data_spec["n_points"],
            original_dimension=data_spec["original_dimension"],
            primary_goal=primary_goal,
            difficulty_summary=context.get("agent2_output", {}).get("difficulty_classification", "N/A"),
            T_min=patch_cfg.get("reliability_threshold_T", 0.90),
            C_min=patch_cfg.get("reliability_threshold_C", 0.90),
            S_max=patch_cfg.get("reliability_threshold_S", 0.30),
            grid_search_results=context.get("grid_search_results", "Not yet computed"),
        )

    def system_message(self) -> str:
        return (
            "You are a specialized agent that estimates maximal embeddable neighborhood "
            "radii for dimensionality reduction. You analyze grid-search reliability curves "
            "and recommend optimal patch sizes. Respond with valid JSON only."
        )
