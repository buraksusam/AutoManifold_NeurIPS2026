"""
Worker Agent 1 — Data Characterization Agent (ENHANCED)

DR_Template Section: Prerequisite for all of Section 1.
Enhanced for Novel Algorithm Generation:
  - Analyzes HD dataset's geometry, density, class imbalance
  - Recommends preprocessing
  - **NEW** Recommends algorithmic primitives (affinity, cost) based on data properties
  - Suggests initial DR methods and hyperparameters
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Data Characterization Agent. Analyze the high-dimensional dataset and recommend algorithmic primitives best suited to its characteristics.

INPUT:
- n = {n_points} data points in D = {original_dimension}-dimensional space
- Labels: {label_summary} (null if unsupervised)
- Design goal: "{primary_goal}"
- Available affinity primitives (grouped by family):
    * SMOOTH DENSITY-ADAPTIVE  : perplexity_gaussian, umap_fuzzy   (preferred HD affinity
      for standard t-SNE / UMAP-style local-structure preservation)
    * RANK-DISCRETE            : mutual_knn, snn, lof_weighted     (used as auxiliary
      blends or for variable-density robustness — not as sole HD affinity for "local" goal)
    * GLOBAL / MANIFOLD        : geodesic, euclidean_knn, learnable
- Available cost primitives: kl_divergence, cross_entropy, spectral, force_directed, mutual_information

TASKS:
1. Estimate the intrinsic dimensionality of the data (e.g., via MLE, correlation dimension, or PCA explained variance).
2. Characterize density: global sparsity level, local density variation, and estimated outlier percentage.
3. Assess geometric structure: suspected manifold type, cluster separability (quantified 0–1), curvature complexity.
4. Identify class imbalance issues. The paper demonstrates that dominant classes (e.g., Proximal Tubule at ~75% of cells) can compress the dynamic range of the visual space and lower label-based separation. Flag any class exceeding 50% of data.
5. Recommend preprocessing steps. The paper uses PCA denoising as a tunable preprocessing step (n_pcs parameter). Suggest an initial n_pcs value based on explained variance analysis.
6. **NEW** Recommend affinity primitives based on data properties. ALWAYS propose a
   primary candidate from the SMOOTH DENSITY-ADAPTIVE family first when primary_goal
   contains "local"; rank-discrete primitives may appear as SECONDARY (auxiliary /
   blend) candidates only.
   - For discrete clustered data (images, digits, categorical): **perplexity_gaussian**
     (primary) — per-point calibrated Gaussian matches the t-SNE recipe that is
     state-of-the-art on Trustworthiness@k for this regime. SNN or mutual_knn may
     appear as low-weight auxiliaries for boundary robustness.
   - For continuous / lineage manifolds (single-cell, trajectory data):
     **umap_fuzzy** (primary) — locally-connected fuzzy-simplicial affinity preserves
     continuous geometry better than rank-discrete primitives.
   - For high curvature manifolds: geodesic (or umap_fuzzy + PCA-denoised distance)
   - For variable density (auxiliary only): lof_weighted
   - For simple structure + global goal: euclidean_knn is sufficient
   Do NOT propose snn / mutual_knn / lof_weighted as the *sole* primary affinity when
   primary_goal contains "local" — they are rank-discrete and bound Trustworthiness@k
   from above. Pair them with a smooth density-adaptive primary instead.
7. **NEW** Recommend cost primitives based on design goal:
   - For local structure focus: kl_divergence, cross_entropy
   - For global structure focus: spectral, force_directed
   - For balanced: multi-objective combinations

DATA SUMMARY PROVIDED:
{data_summary}

DATA SUMMARY PROVIDED:
{data_summary}

AVAILABLE PRIMITIVES REFERENCE:
AFFINITY PRIMITIVES (listed primary first within each family):
  SMOOTH DENSITY-ADAPTIVE:
  - perplexity_gaussian: t-SNE style per-point calibrated Gaussian. σ_i chosen by
    binary search so Shannon entropy of P(.|i) = log2(perplexity). Best-in-class for
    Trustworthiness@k on clustered / categorical data. Pair with student_t(dof=1)
    LD kernel + KL cost for the t-SNE recipe.
  - umap_fuzzy: UMAP locally-connected fuzzy-simplicial affinity. Shift by ρ_i
    (nearest neighbor distance) and scale by σ_i so Σ_j exp(-max(0, d_ij-ρ_i)/σ_i)
    = log2(k). Best for continuous / lineage manifolds. Pair with umap_ab LD kernel +
    cross-entropy cost for the UMAP recipe.
  RANK-DISCRETE (auxiliary / blend only for "local" goal):
  - mutual_knn: Mutual k-nearest neighbors (both directions). Robust to outliers.
  - snn: Shared nearest neighbors. Handles variable density; good for cluster robustness.
  - lof_weighted: LOF-weighted affinities. Density-aware, auxiliary for sparse regions.
  GLOBAL / MANIFOLD:
  - geodesic: Manifold geodesic distances (Isomap-style). Good for curved manifolds.
  - euclidean_knn: Standard k-nearest neighbors with Euclidean distance.
  - learnable: Parametric/neural network learned metric.

COST PRIMITIVES:
- kl_divergence: t-SNE style KL(P||Q). Preserves local neighborhoods, may lose global structure.
- cross_entropy: UMAP-style cross-entropy with negative sampling. Scalable, balances local/global.
- spectral: Spectral embedding loss. Preserves global structure, less emphasis on local.
- force_directed: Force-directed layout (spring-electrical). Interpretable, visual.
- mutual_information: Information-maximization. Preserves statistical dependencies.

REQUIRED OUTPUT (JSON):
{{
  "data_properties": {{
    "intrinsic_dimensionality_estimate": float,
    "explained_variance_by_pcs": {{"10": float, "20": float, "50": float}},
    "density_characteristics": {{
      "global_sparsity": "sparse | moderate | dense",
      "local_density_variation": "low | moderate | high",
      "estimated_outlier_percentage": float
    }},
    "geometric_features": {{
      "suspected_manifold_structure": "string",
      "cluster_separability_score": float,
      "curvature_complexity": "low | moderate | high"
    }},
    "class_imbalance": {{
      "present": boolean,
      "dominant_class_percentage": float,
      "imbalance_severity": "none | mild | severe",
      "impact_on_visualization": "string"
    }}
  }},
  "recommendations": {{
    "preprocessing": {{
      "pca_denoising": boolean,
      "suggested_n_pcs": int,
      "other_steps": ["string"]
    }},
    "primitive_recommendations": {{
      "affinity_candidates": [
        {{
          "primitive": "string",
          "rationale": "string",
          "best_for_regions": ["H", "M", "E"],
          "complexity": "low | medium | high"
        }}
      ],
      "cost_candidates": [
        {{
          "primitive": "string",
          "rationale": "string",
          "addresses": ["local_preservation", "global_structure", "both"]
        }}
      ]
    }},
    "initial_dr_suggestions": [
      {{
        "method": "string",
        "rationale": "string",
        "suggested_initial_hyperparameters": {{}}
      }}
    ],
    "potential_challenges": ["string"]
  }},
  "diagnostic_questions": ["string"]
}}"""


class DataCharacterizationAgent(BaseAgent):
    AGENT_NAME = "DataCharacterization"
    AGENT_ID = 1

    def build_prompt(self, context: dict) -> str:
        data_spec = context["data_spec"]

        # Pre-compute summary statistics to include in prompt
        data_summary = context.get("data_summary", "Not provided — use the data_spec to reason.")

        return PROMPT_TEMPLATE.format(
            n_points=data_spec["n_points"],
            original_dimension=data_spec["original_dimension"],
            label_summary=data_spec.get("label_summary", "null"),
            primary_goal=context["design_constraints"]["primary_goal"],
            data_summary=data_summary,
        )

    def system_message(self) -> str:
        return (
            "You are a specialized data scientist analyzing high-dimensional datasets "
            "for novel dimensionality reduction algorithm design. You must characterize data geometry, "
            "density, class balance, and recommend algorithmic primitives (affinity, cost functions). "
            "Respond with valid JSON only."
        )
