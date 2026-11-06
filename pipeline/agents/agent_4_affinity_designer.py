"""
Worker Agent 4 — Affinity Graph Designer Agent (NEW)

Purpose for Novel Algorithm Generation:
  - Designs the neighborhood/affinity graph structure
  - Key source of algorithmic novelty
  - Selects and composes affinity primitives
  - Specifies neighborhood sizes per difficulty region
"""

from ..base_agent import BaseAgent


def _strip_between(text: str, start_marker: str, end_marker: str, replacement: str = "") -> str:
    """Remove the prompt block from `start_marker` up to (not including)
    `end_marker`, optionally leaving a short note. Used by the component
    ablation to drop the primitive-library / composition-pattern sections."""
    i = text.find(start_marker)
    if i == -1:
        return text
    j = text.find(end_marker, i)
    if j == -1:
        return text
    note = (replacement + "\n\n") if replacement else ""
    return text[:i] + note + text[j:]


def _format_baseline_section(profile: dict | None) -> str:
    """Render baseline_profile as a compact agent-readable block."""
    if not profile:
        return "  (no baseline calibration available)"
    winner = profile.get("winner", {}).get("name", "?")
    score  = profile.get("winner", {}).get("primary_goal_score", "?")
    intent = profile.get("intent", "?")
    anchor = profile.get("recipe_anchor", {})
    targets = profile.get("beat_targets", {})
    weakness = profile.get("weakness_analysis", {})
    per_method = profile.get("metrics_per_method", {})

    tol = targets.get("tolerance", "?")
    metric_lines = []
    for k, v in targets.items():
        if k == "tolerance":
            continue
        metric_lines.append(f"    {k} = {v}")
    metrics_block = "\n".join(metric_lines) if metric_lines else "    (none reported)"

    lines = [
        f"  Winner baseline (by primary_goal='{intent}'): {winner}  (score={score})",
        f"  What worked   : family={anchor.get('family','?')}  "
        f"hd_affinity={anchor.get('hd_affinity','?')}  "
        f"ld_kernel={anchor.get('ld_kernel',{}).get('primitive','?')}  "
        f"cost={anchor.get('cost','?')}",
        f"  Metrics to beat (tol={tol}, direction governed by primary_goal):",
        metrics_block,
        f"  Winner weakness: {weakness.get('winner_weakest_metric','?')}  "
        f"(runner-up wins: {weakness.get('runner_up_wins') or 'none'})",
        f"  Augmentation focus: {weakness.get('augmentation_focus','')}",
        f"  Per-method metrics: {per_method}",
        "",
        "  → INFORMATIONAL: this tells you which family of structure the data rewards "
        "and what numbers your novel design needs to clear. It is NOT a prescription. "
        "Use it as evidence when choosing an affinity family, and feel free to deviate "
        "when Agent 1 / Agent 2 signals a different structure — just justify the choice "
        "in design_rationale.",
    ]
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are the Affinity Graph Designer. Design how neighbors are defined in the high-dimensional space — this is a key source of algorithmic novelty.

DESIGN INTENT (primary_goal): {primary_goal}

This intent governs which affinity family you must treat as the default:
  - "preserve local structure" / "local"   → default primary = perplexity_gaussian or umap_fuzzy
  - "reveal global topology"  / "global"   → default primary = geodesic or a spectral-compatible primitive
  - "balance both" / "balanced"            → Pattern A hybrid with a smooth density-adaptive
                                             primitive carrying >= 0.6 of the primary weight
  - EXPLICIT SINGLE-METRIC GOALS (if the primary_goal text names a specific metric):
      * "maximum trustworthiness" / "maximum continuity" / "maximum LOF preservation"
          → treat as a STRICT local goal; primary MUST come from the smooth
            density-adaptive family (perplexity_gaussian or umap_fuzzy); any
            rank-discrete / global primitive may appear only as low-weight
            auxiliary (≤ 0.15).
      * "maximum silhouette"
          → treat as a local goal with emphasis on cluster separability; prefer
            perplexity_gaussian as primary (t-SNE-style attraction produces tighter
            clusters).
      * "minimum stress" / "maximum spearman"
          → treat as a STRICT global goal; primary MUST be geodesic, euclidean_knn,
            or a spectral-compatible primitive; do NOT pick a rank-discrete primary.

  If the primary_goal is an explicit single-metric goal, every design choice must
  be justified by how it raises/lowers THAT metric; other metrics are secondary.

You may override the default ONLY IF the data properties (Agent 1) or difficulty mapping
(Agent 2) provide a concrete reason the default family is unsafe for THIS dataset.
State the reason explicitly in `design_rationale` if you override.

PRIMITIVE FAMILIES:
  [smooth, density-adaptive]  perplexity_gaussian, umap_fuzzy
  [rank-discrete / binary]    euclidean_knn, mutual_knn, snn, lof_weighted
  [global / manifold]         geodesic, learnable

WITHIN THE SMOOTH DENSITY-ADAPTIVE FAMILY — which one to pick?

Both `perplexity_gaussian` and `umap_fuzzy` are smooth density-adaptive, but they are
NOT interchangeable. For primary_goal = "local", the choice between them controls the
Trustworthiness(k) / Continuity(k) ceiling.

- Prefer `perplexity_gaussian` (t-SNE family) when Agent 1's data_properties indicate:
    * Discrete, well-separated clusters (high baseline Silhouette, strong modal separation)
    * Tabular / categorical / image-patch / handwritten / spectral data
    * No evident continuous manifold or lineage structure
    * Clear class boundaries with low inter-class connectivity
  Recipe: perplexity_gaussian + student_t(dof=1) + kl_divergence
  Rationale: full perplexity-calibrated Gaussian with symmetrization
      p_ij = (p_{{j|i}} + p_{{i|j}}) / (2n)
  gives the tightest rank preservation around each point; the heavy-tailed Student-t
  LD kernel resolves crowding. This recipe empirically maximises T@k on
  discretely-clustered data.

- Prefer `umap_fuzzy` (UMAP family) when Agent 1's data_properties indicate:
    * Continuous manifold structure, lineage trajectories, curved connectivity
    * Single-cell / developmental / flow-cytometry / trajectory data
    * Graph-like or relational data with transitive connectivity
    * Emphasis on global layout / distance-scale preservation alongside local
  Recipe: umap_fuzzy + umap_ab(a, b) + cross_entropy
  Rationale: the locally-connected rho_i shift plus fuzzy-union symmetrization retains
  continuous manifold topology and typically gives better global Spearman, at a small
  cost in maximum T@k versus the t-SNE recipe.

TIE-BREAKERS when signals are mixed or ambiguous:
  - primary_goal = "local"    → prefer perplexity_gaussian (higher T@k ceiling in isolation)
  - primary_goal = "balanced" → prefer umap_fuzzy (better global/local trade-off)
  - primary_goal = "global"   → these are rarely primary; use geodesic/spectral instead

CONTEXT:
- Data characteristics: {data_properties}
- Difficulty mapping: {difficulty_primitive_mapping}
- Composition opportunities: {composition_opportunities}
- Available affinity primitives: {affinity_primitives}

BASELINE CALIBRATION (Phase 0 — SOFT BIAS, may be overridden with explicit justification):
{baseline_section}

COMPOSITION PATTERNS (Reference):

Pattern A: Hybrid Affinity
  affinity_p(x, y) = α · affinity_A(x, y) + β · affinity_B(x, y)
  Example: Mutual k-NN + SNN
    affinity(x, y) = 0.7 · mutual_knn(x, y) + 0.3 · snn(x, y)
    → Combines robustness with scalability

Pattern C: Adaptive Weighting
  λ_i = f(local_density(x_i), curvature(x_i), difficulty_i)
  Example: Density-adaptive local weight
    λ_local(x_i) = 1.0 - normalize(density(x_i))
    → Stronger local preservation in sparse regions

AFFINITY PRIMITIVES REFERENCE:
- euclidean_knn: Standard k-nearest neighbors with Euclidean distance
  - Good for: Simple geometries, baseline comparisons
  - Complexity: Low
  - Limitations: Sensitive to outliers, assumes Euclidean structure

- mutual_knn: Mutual k-nearest neighbors (both directions)
  - Good for: Clustered data, robust to outliers
  - Complexity: Low
  - Limitations: May break for continuous manifolds

- snn: Shared nearest neighbors
  - Good for: Variable density, very robust
  - Complexity: Medium
  - Limitations: Higher computational cost

- geodesic: Manifold geodesic distances (Isomap-style)
  - Good for: Curved manifolds, global structure
  - Complexity: High
  - Limitations: Sensitive to shortcuts, expensive for large n

- lof_weighted: LOF-weighted affinities
  - Good for: Density-aware, handles sparse regions
  - Complexity: Medium
  - Limitations: Requires LOF precomputation

- learnable: Parametric/neural network learned metric
  - Good for: Complex non-Euclidean structure
  - Complexity: Very High
  - Limitations: Requires training, less interpretable

- perplexity_gaussian: Per-point perplexity-calibrated Gaussian conditional affinity (the HD affinity used inside t-SNE)
  - Formula:
      p_{{j|i}} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2)) / Z_i         (Z_i normalizes over j != i)
      sigma_i is chosen by binary search so that the Shannon entropy of p_{{.|i}}
        equals log2(perplexity), i.e. the effective neighborhood size is fixed.
      Symmetrize: p_ij = (p_{{j|i}} + p_{{i|j}}) / (2 * n)
  - Good for: Local neighborhood preservation (maximises Trustworthiness / Continuity).
      sigma_i adapts to local density automatically, so the affinity is a smooth,
      density-aware probability rather than a binary adjacency.
  - Complexity: Medium (binary search per point, O(n * k * log(1/eps)))
  - Limitations: Full pairwise cost without approximation; perplexity must be tuned.
  - Implementation note: do NOT import sklearn.manifold._t_sne internals; construct
      p_{{j|i}} explicitly from a kNN distance matrix (k ~ 3 * perplexity is sufficient).

- umap_fuzzy: UMAP-style fuzzy-simplicial local membership (the HD affinity used inside UMAP)
  - Formula:
      rho_i = min over j in NN(i), j != i, of ||x_i - x_j||        (nearest-neighbor distance)
      sigma_i solves:  sum_j exp(-(||x_i - x_j|| - rho_i)_+ / sigma_i) = log2(k)
        where (a)_+ = max(a, 0) and j ranges over the k nearest neighbours of i.
      Directed edge: p_{{j|i}} = exp(-(||x_i - x_j|| - rho_i)_+ / sigma_i)
      Fuzzy-union symmetrize: p_ij = p_{{j|i}} + p_{{i|j}} - p_{{j|i}} * p_{{i|j}}
  - Good for: Local structure with good global placement; locally-connected manifold
      assumption makes isolated points attach to their nearest neighbour.
  - Complexity: Medium (binary search per point, same order as perplexity_gaussian).
  - Limitations: Parameter k controls local vs global trade-off and must be tuned.
  - Implementation note: do NOT call the `umap` library's internal utilities; construct
      directly from a sparse kNN graph with the formula above.

TASKS:
1. Read DESIGN INTENT above and pick the *primary* affinity from the appropriate family.

   Capacity note for local-preservation goals:
     Rank-discrete primitives (euclidean_knn, mutual_knn, snn, lof_weighted) discard the
     continuous similarity-vs-distance mapping within the k-NN set. Trustworthiness(k)
     and Continuity(k) measure rank preservation of continuous neighbourhoods; a pipeline
     whose primary HD affinity is rank-discrete therefore cannot match the local-metric
     ceiling reached by a smooth density-adaptive primary (perplexity_gaussian,
     umap_fuzzy). For primary_goal = "local", a rank-discrete primary is a capacity
     ceiling, not a novelty gain.

2. If you hybridize (Pattern A), enforce the BLEND SAFETY RULE:
     - primary_goal = "local":
         smooth density-adaptive component weight >= 0.85
         rank-discrete components (snn, mutual_knn, lof_weighted) combined weight <= 0.15
         Rationale: rank-discrete blends dilute T@k by discarding continuous similarity
         within the k-NN set. For local preservation, keep the smooth primary near-pure
         and reserve rank-discrete mass for targeted robustness in a small sub-region.
     - primary_goal = "balanced":
         smooth density-adaptive component weight >= 0.6
         rank-discrete components combined weight <= 0.4
     - primary_goal = "global":
         smooth density-adaptive components may be demoted below 0.5 in favour of
         geodesic / global primitives.

   If you use adaptive per-point weighting (e.g. w_i = f(density_i, difficulty_i)),
   the SPATIAL MINIMUM of the smooth component's weight across all points must still
   satisfy the floor above. A scheme that drops the smooth weight to 0.6 in H regions
   violates the "local" floor of 0.85.

3. Design adaptive weighting only AFTER choosing the primary. Adaptive schemes modulate
   the *blend ratio*, not the primary primitive itself.

4. Specify neighborhood sizes per region (H/M/E). Local metrics are evaluated at fixed
   k = 10; oversizing k in H regions buys robustness, not T@10 / C@10.

5. Define computational approach and expected properties.

REQUIRED OUTPUT (JSON):
{{
  "affinity_design": {{
    "primary_primitive": "string",
    "is_hybrid": boolean,
    "hybrid_components": [
      {{
        "primitive": "string",
        "weight": "float or 'adaptive'",
        "responsibility": "string (e.g., 'robustness_to_noise', 'global_structure')"
      }}
    ],
    "neighborhood_specification": {{
      "H_regions": {{"type": "knn | radius | adaptive", "value": "int or float or 'auto'"}},
      "M_regions": {{"type": "...", "value": "..."}},
      "E_regions": {{"type": "...", "value": "..."}}
    }},
    "adaptive_weighting": {{
      "enabled": boolean,
      "weight_function": "string (e.g., 'density_based', 'curvature_based', 'difficulty_based')",
      "formula": "string (mathematical expression if applicable)"
    }}
  }},
  "expected_properties": {{
    "robustness_to_noise": "low | medium | high",
    "computational_cost": "low | medium | high",
    "manifold_sensitivity": "low | medium | high",
    "density_aware": boolean
  }},
  "implementation_notes": {{
    "key_libraries": ["string (e.g., 'scikit-learn', 'pynndescent', 'hnswlib')"],
    "approximation_methods": ["string (e.g., 'NN-descent', 'HNSW', 'ball tree')"],
    "precomputation_requirements": ["string"]
  }},
  "design_rationale": "string (2-3 sentences explaining why this design fits the data)"
}}"""


class AffinityDesignerAgent(BaseAgent):
    """Designs the affinity/neighborhood graph structure for novel DR algorithms."""

    AGENT_NAME = "AffinityDesigner"
    AGENT_ID = 4

    def build_prompt(self, context: dict) -> str:
        agent1_output = context.get("agent1_output", {})
        agent2_output = context.get("agent2_output", {})
        available_methods = context.get("available_methods", {})
        design_constraints = context.get("design_constraints", {})
        primary_goal = design_constraints.get("primary_goal", "balance both")

        data_properties = agent1_output.get("data_properties", {})
        difficulty_mapping = agent2_output.get("difficulty_primitive_mapping", {})
        composition_opportunities = agent2_output.get("composition_opportunities", [])

        disabled = set(context.get("disabled_components", []))

        # Get affinity primitives from context or use defaults
        affinity_primitives = available_methods.get("affinity_primitives", [
            "euclidean_knn", "mutual_knn", "snn", "geodesic", "lof_weighted", "learnable",
            "perplexity_gaussian", "umap_fuzzy",
        ])
        affinity_primitives_str = (
            "(none provided — design the affinity graph from first principles)"
            if "primitive_library" in disabled else str(affinity_primitives)
        )

        baseline_section = _format_baseline_section(context.get("baseline_profile"))

        prompt = PROMPT_TEMPLATE.format(
            primary_goal=primary_goal,
            data_properties=str(data_properties),
            difficulty_primitive_mapping=str(difficulty_mapping),
            composition_opportunities=str(composition_opportunities),
            affinity_primitives=affinity_primitives_str,
            baseline_section=baseline_section,
        )
        # Ablation: strip the primitive-library / composition-pattern prompt blocks.
        if "patterns" in disabled:
            prompt = _strip_between(prompt, "COMPOSITION PATTERNS (Reference):",
                                    "AFFINITY PRIMITIVES REFERENCE:",
                                    "[composition-pattern guidance removed for ablation]")
        if "primitive_library" in disabled:
            prompt = _strip_between(prompt, "AFFINITY PRIMITIVES REFERENCE:", "TASKS:",
                                    "[affinity primitive library removed for ablation]")
            prompt = _strip_between(prompt, "PRIMITIVE FAMILIES:", "CONTEXT:",
                                    "[primitive families removed for ablation]")
        return prompt

    def system_message(self) -> str:
        return (
            "You are a specialized agent that designs affinity/neighborhood graphs "
            "for novel dimensionality reduction algorithms. You select and compose "
            "affinity primitives, specify neighborhood sizes, and define adaptive weighting. "
            "Respond with valid JSON only."
        )

    def _validate(self, output: dict) -> None:
        """Validate that required fields are present."""
        required_sections = ["affinity_design", "expected_properties", "implementation_notes"]
        for section in required_sections:
            if section not in output:
                self.logger.warning(f"[{self.AGENT_NAME}] Missing required section: {section}")

        # Validate affinity_design structure
        if "affinity_design" in output:
            affinity = output["affinity_design"]
            required_keys = ["primary_primitive", "is_hybrid", "neighborhood_specification"]
            for key in required_keys:
                if key not in affinity:
                    self.logger.warning(f"[{self.AGENT_NAME}] Missing affinity_design key: {key}")
