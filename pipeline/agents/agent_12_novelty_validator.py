"""
Worker Agent 12 — Novelty Validator Agent (NEW)

Purpose for Novel Algorithm Generation:
  - Assesses whether the composed algorithm is genuinely novel
  - Checks for mathematical equivalence with existing methods
  - Classifies novelty level (incremental, compositional, fundamental)
  - Prevents "reinventing" existing algorithms
"""

from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Novelty Validator. Assess whether the designed algorithm is genuinely novel or equivalent to existing methods.

CONTEXT:
- Affinity design: {affinity_design}
- Cost design: {cost_design}
- Optimization design: {optimization_design}

KNOWN EXISTING METHODS (for equivalence checking):

t-SNE (van der Maaten & Hinton, 2008):
  - Affinity: Conditional probability p_j|i = exp(-||x_i - x_j||^2 / 2σ_i^2) / Σ exp(-||x_i - x_k||^2 / 2σ_i^2)
  - Cost: KL divergence KL(P||Q) = Σ p_ij log(p_ij / q_ij)
  - Optimization: Gradient descent with early exaggeration

UMAP (McInnes et al., 2018):
  - Affinity: Mutual k-NN + fuzzy simplicial set (weighted edges)
  - Cost: Cross-entropy with negative sampling: -Σ [p_ij log(q_ij) + (1-p_ij) log(1-q_ij)]
  - Optimization: Stochastic gradient descent

PaCMAP (Wang et al., 2021):
  - Affinity: Weighted nearest neighbors with MN/FP pair sampling
  - Cost: Cross-entropy variant with pair types
  - Optimization: Adam with three-phase learning rate schedule

LargeVis (Tang et al., 2016):
  - Affinity: k-NN graph
  - Cost: Negative sampling cross-entropy
  - Optimization: Async SGD

Spectral Methods (e.g., Laplacian Eigenmaps):
  - Affinity: Various (often Gaussian kernel)
  - Cost: Tr(Y^T L Y) where L is graph Laplacian
  - Optimization: Eigendecomposition

ForceAtlas2 (Jacomy et al., 2014):
  - Affinity: Graph edges
  - Cost: Force-directed (repulsion + attraction)
  - Optimization: Iterative force updates

TASKS:
1. Identify which SINGLE existing method is *most similar* to the designed algorithm
   (the nearest reference — e.g., t-SNE, UMAP, PaCMAP, LargeVis, Laplacian Eigenmaps,
   ForceAtlas2). Pick one — the closest match, not a list.
2. Explain HOW the designed algorithm *theoretically differs* from that nearest method,
   specifically in terms of what it does differently *to/with the data* — how it
   transforms high-dimensional similarities, where it puts gradient pressure, what
   aspect of structure it prioritises. Be brief but precise (2–4 sentences,
   mechanism-level, not marketing).
3. Suggest a concise name for the designed algorithm.

REQUIRED OUTPUT (JSON):
{{
  "most_similar_method": "string (single nearest existing method, e.g. 't-SNE', 'UMAP', 'PaCMAP', 'Laplacian Eigenmaps')",
  "theoretical_differences": "string (2–4 sentences; precise, mechanism-level explanation of how THIS algorithm handles the data differently than `most_similar_method` — e.g. 'Where t-SNE calibrates a single per-point Gaussian perplexity, this design adds a locally-connected rho_i shift before sigma calibration and blends the symmetrised graph with a mutual-kNN mask weighted by local density; in practice this pulls low-density points more aggressively toward their nearest dense neighbours and lets boundary points drift less than t-SNE would.')",
  "suggested_name": "string (e.g., 'Density-Adaptive Spectral t-SNE')",
  "tagline": "string (one-line description of the novel aspect)",
  "is_genuinely_novel": boolean,
  "recommendations": {{
    "next_steps": ["string"],
    "potential_limitations": ["string"],
    "comparison_baselines": ["string (which methods to compare against in evaluation)"]
  }}
}}"""


class NoveltyValidatorAgent(BaseAgent):
    """Validates whether a designed DR algorithm is genuinely novel."""

    AGENT_NAME = "NoveltyValidator"
    AGENT_ID = 12

    def build_prompt(self, context: dict) -> str:
        agent4_output = context.get("agent4_output", {})
        agent5_output = context.get("agent5_output", {})
        agent6_output = context.get("agent6_output", {})

        affinity_design = agent4_output.get("affinity_design", {})
        cost_design = agent5_output.get("cost_design", {})
        optimization_design = agent6_output.get("optimization_design", {})

        return PROMPT_TEMPLATE.format(
            affinity_design=str(affinity_design),
            cost_design=str(cost_design),
            optimization_design=str(optimization_design),
        )

    def system_message(self) -> str:
        return (
            "You are a specialized agent that validates the novelty of dimensionality "
            "reduction algorithms. You check for mathematical equivalence with existing "
            "methods (t-SNE, UMAP, PaCMAP, etc.) and classify the level of novelty. "
            "Be rigorous in detecting equivalence — small parameter changes are NOT novel. "
            "Respond with valid JSON only."
        )

    def _validate(self, output: dict) -> None:
        """Validate that required fields are present."""
        required_keys = ["most_similar_method", "theoretical_differences", "suggested_name"]
        for key in required_keys:
            if not output.get(key):
                self.logger.warning(f"[{self.AGENT_NAME}] Missing or empty key: {key}")
