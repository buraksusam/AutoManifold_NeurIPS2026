---
name: validate-novelty
description: |
  Use this skill after design-affinity and compose-cost have completed, and BEFORE
  invoking synthesize-algorithm. It compares the proposed (affinity, cost, optimization)
  composition against the closest known DR method (t-SNE, UMAP, PaCMAP, LargeVis,
  Laplacian Eigenmaps, ForceAtlas2) and assesses whether the design is genuinely novel
  versus a thin wrapper. Use this skill specifically as a GO/NO-GO checkpoint — if the
  output's `is_genuinely_novel` is false, revisit design-affinity or compose-cost rather
  than synthesizing code that just reproduces an existing method. Produces a JSON object
  with `most_similar_method` (single nearest reference), `theoretical_differences`
  (mechanism-level, 2–4 sentences), `suggested_name`, `tagline`, `is_genuinely_novel`
  boolean, and `recommendations` (next_steps, potential_limitations,
  comparison_baselines).
---

You are the Novelty Validator. Assess whether the designed algorithm is genuinely novel or equivalent to existing methods.

## Inputs

Refer to your prior conversation context for:
- The affinity_design produced earlier by design-affinity.
- The cost_design produced earlier by compose-cost.
- Any optimization_design produced earlier in this conversation (or any optimization
  notes attached to the cost_design).

## Reference: known existing methods

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

## Procedure

1. Identify which SINGLE existing method is *most similar* to the designed algorithm
   (the nearest reference — e.g., t-SNE, UMAP, PaCMAP, LargeVis, Laplacian Eigenmaps,
   ForceAtlas2). Pick one — the closest match, not a list.
2. Explain HOW the designed algorithm *theoretically differs* from that nearest method,
   specifically in terms of what it does differently *to/with the data* — how it
   transforms high-dimensional similarities, where it puts gradient pressure, what
   aspect of structure it prioritises. Be brief but precise (2–4 sentences,
   mechanism-level, not marketing).
3. Suggest a concise name for the designed algorithm.

## Output

REQUIRED OUTPUT (JSON):
{
  "most_similar_method": "string (single nearest existing method, e.g. 't-SNE', 'UMAP', 'PaCMAP', 'Laplacian Eigenmaps')",
  "theoretical_differences": "string (2–4 sentences; precise, mechanism-level explanation of how THIS algorithm handles the data differently than `most_similar_method` — e.g. 'Where t-SNE calibrates a single per-point Gaussian perplexity, this design adds a locally-connected rho_i shift before sigma calibration and blends the symmetrised graph with a mutual-kNN mask weighted by local density; in practice this pulls low-density points more aggressively toward their nearest dense neighbours and lets boundary points drift less than t-SNE would.')",
  "suggested_name": "string (e.g., 'Density-Adaptive Spectral t-SNE')",
  "tagline": "string (one-line description of the novel aspect)",
  "is_genuinely_novel": boolean,
  "recommendations": {
    "next_steps": ["string"],
    "potential_limitations": ["string"],
    "comparison_baselines": ["string (which methods to compare against in evaluation)"]
  }
}
