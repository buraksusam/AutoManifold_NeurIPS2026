"""
Worker Agent 2 — Embedding Difficulty Analyzer Agent (ENHANCED)

DR_Template Section 1.1: Identify Embedding Difficulty Level.
Enhanced for Novel Algorithm Generation:
  - Divides data into ~100-point patches
  - Runs t-SNE/UMAP per patch
  - Classifies into H/M/E difficulty groups
  - **NEW** Maps each difficulty group to recommended algorithmic primitives
  - Identifies opportunities for hybrid compositions
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Embedding Difficulty Analyzer Agent. Classify data regions into H/M/E groups and map each to recommended algorithmic primitives.

CONTEXT (from DR_Template Section 1.1):
"Divide the n data points into groups that are 'hard-to-embed' (H), 'moderately difficult to embed' (M), and 'easy-to-embed' (E) when being mapped into a low-dimensional space with d = 2. Higher embedding reliability scores indicate less difficulty level to embed."

INPUT:
- Data: n = {n_points}, D = {original_dimension}
- Preprocessing recommendations: {preprocessing}
- Primitive recommendations from Agent 1: {primitive_recommendations}
- Available clustering methods: {clustering_algorithms}

PATCH-LEVEL RELIABILITY SCORES (pre-computed):
{patch_scores}

PROCEDURE (from DR_Template):
Approach 1 (Primary):
  1. Data divided into local patches of ~m ≈ 100 points using clustering.
  2. t-SNE and UMAP run on each patch independently.
  3. Reliability scores computed: Trustworthiness T(k), Continuity C(k), Stress S.
  4. Aggregate score per patch = avg(T, C, 1−S) across both methods.
  5. Set thresholds to partition into H / M / E.

**NEW - Primitive Mapping:**
  Your per-region recommendations are BLEND-WEIGHT guidance, not a replacement for
  the primary HD affinity. Agent 4 will pick ONE smooth density-adaptive primary
  (perplexity_gaussian or umap_fuzzy) that applies globally; the H/M/E mapping here
  suggests how much auxiliary rank-discrete support each region benefits from.
  - H regions (hard to embed): highest SNN / mutual_knn / lof_weighted auxiliary
    weight for cluster-boundary robustness (≤ 0.15 blend weight).
  - M regions: small auxiliary weight, mostly rely on the smooth primary.
  - E regions: no auxiliary needed; pure smooth primary suffices.
  - Identify composition opportunities (adaptive per-point weights, density floors).
  Do NOT recommend `snn`, `mutual_knn`, `lof_weighted`, or `euclidean_knn` as the
  sole primary affinity — these are rank-discrete or isotropic and bound
  Trustworthiness@k below the t-SNE / UMAP baseline.

The patch-level scores above have already been computed. Your task is to:
  1. Analyze the score distribution and set appropriate thresholds.
  2. Classify patches into H, M, E groups.
  3. **NEW** Map each difficulty group to recommended affinity primitives.
  4. **NEW** Identify composition opportunities (e.g., density-weighted affinity, adaptive schemes).
  5. Characterize each group and recommend a data feeding strategy.

REQUIRED OUTPUT (JSON):
{{
  "methodology": {{
    "approach_used": "Approach 1 | Approach 2 | Hybrid",
    "patch_creation": {{
      "clustering_method": "string",
      "target_patch_size_m": 100,
      "n_patches_created": int,
      "actual_patch_size_distribution": {{"min": int, "mean": float, "max": int}}
    }}
  }},
  "difficulty_classification": {{
    "thresholds": {{
      "threshold_H": float,
      "threshold_E": float,
      "threshold_rationale": "string"
    }},
    "groups": {{
      "H": {{
        "n_patches": int,
        "patch_indices": [int],
        "percentage_of_data": float,
        "avg_trustworthiness": float,
        "avg_continuity": float,
        "avg_stress": float,
        "characteristics": "string"
      }},
      "M": {{
        "n_patches": int,
        "patch_indices": [int],
        "percentage_of_data": float,
        "avg_trustworthiness": float,
        "avg_continuity": float,
        "avg_stress": float,
        "characteristics": "string"
      }},
      "E": {{
        "n_patches": int,
        "patch_indices": [int],
        "percentage_of_data": float,
        "avg_trustworthiness": float,
        "avg_continuity": float,
        "avg_stress": float,
        "characteristics": "string"
      }}
    }}
  }},
  "difficulty_primitive_mapping": {{
    "H_regions": {{
      "auxiliary_affinity": "string (e.g., 'snn', 'mutual_knn', 'lof_weighted')",
      "auxiliary_blend_weight": "float in [0, 0.15] — max auxiliary weight in this region",
      "justification": "string (why this auxiliary helps hard regions)",
      "fallback_options": ["string"]
    }},
    "M_regions": {{
      "auxiliary_affinity": "string (e.g., 'snn', 'none')",
      "auxiliary_blend_weight": "float in [0, 0.10]",
      "justification": "string",
      "fallback_options": ["string"]
    }},
    "E_regions": {{
      "auxiliary_affinity": "none",
      "auxiliary_blend_weight": 0.0,
      "justification": "Easy regions need no auxiliary; smooth primary suffices",
      "fallback_options": []
    }}
  }},
  "composition_opportunities": [
    {{
      "opportunity": "string (e.g., 'density-weighted affinity', 'hybrid_snn_mutual_knn')",
      "where": "string (e.g., 'H and M region boundaries', 'transition zones')",
      "expected_benefit": "string"
    }}
  ],
  "spatial_distribution": {{
    "description": "string",
    "H_near_M": boolean,
    "E_contiguous": boolean
  }},
  "design_implications": {{
    "recommended_approach": "unified | stratified | hybrid",
    "rationale": "string"
  }}
}}"""


class EmbeddingDifficultyAgent(BaseAgent):
    AGENT_NAME = "EmbeddingDifficulty"
    AGENT_ID = 2

    def build_prompt(self, context: dict) -> str:
        data_spec = context["data_spec"]
        agent1_output = context.get("agent1_output", {})
        primitive_recommendations = agent1_output.get("recommendations", {}).get("primitive_recommendations", {})

        return PROMPT_TEMPLATE.format(
            n_points=data_spec["n_points"],
            original_dimension=data_spec["original_dimension"],
            preprocessing=agent1_output.get("recommendations", {}).get("preprocessing", "N/A"),
            primitive_recommendations=str(primitive_recommendations) if primitive_recommendations else "Not provided",
            clustering_algorithms=context["available_methods"]["clustering_algorithms"],
            patch_scores=context.get("patch_scores", "Not yet computed"),
        )

    def system_message(self) -> str:
        return (
            "You are a specialized agent that classifies data regions by embedding difficulty "
            "and maps each difficulty group to recommended algorithmic primitives. "
            "You analyze per-patch DR reliability scores, partition into Hard (H), Moderate (M), "
            "and Easy (E) groups, and recommend affinity primitives for each group. "
            "Respond with valid JSON only."
        )
