"""
Worker Agent 5 — Cost Function Composer Agent (NEW)

Purpose for Novel Algorithm Generation:
  - Designs the embedding objective function
  - Key source of algorithmic novelty
  - Selects and composes cost function primitives
  - Specifies local vs global trade-offs
"""

from ..base_agent import BaseAgent


# Static catalog mirroring QuantitativeAnalystEngine.compute_all() output buckets.
# Kept in sync by hand because Agent 5 runs at design time (no embedding to score yet),
# so we expose what metrics WILL be available rather than computing them now.
METRICS_ENGINE_CATALOG = {
    "global": [
        {"name": "Spearman Correlation",  "direction": "higher_is_better",
         "range": "[-1, 1]", "surrogate": "spearman_rank_loss"},
        {"name": "Stress",                "direction": "lower_is_better",
         "range": "[0, inf)", "surrogate": "kruskal_stress"},
        {"name": "Mean Distance Ratio",   "direction": "near_1_is_best",
         "range": "(0, inf)", "surrogate": "mean_distance_ratio"},
    ],
    "local": [
        {"name": "Trustworthiness",       "direction": "higher_is_better",
         "range": "[0, 1]", "surrogate": "knn_precision_loss"},
        {"name": "Continuity",            "direction": "higher_is_better",
         "range": "[0, 1]", "surrogate": "knn_recall_loss"},
    ],
    "cluster": [
        {"name": "Silhouette Score",      "direction": "higher_is_better",
         "range": "[-1, 1]", "surrogate": "silhouette_surrogate"},
    ],
    "outlier": [
        {"name": "LOF Median",            "direction": "near_1_is_best",
         "range": "[0, inf)", "surrogate": "lof_distribution_match"},
        {"name": "LOF Outlier %",         "direction": "lower_is_better",
         "range": "[0, 100]", "surrogate": "lof_count"},
    ],
}


def _format_metrics_catalog(catalog: dict) -> str:
    lines = []
    for bucket, metrics in catalog.items():
        lines.append(f"  [{bucket.upper()} bucket]")
        for m in metrics:
            lines.append(
                f"    - {m['name']:<22} dir={m['direction']:<18} "
                f"range={m['range']:<12} surrogate={m['surrogate']}"
            )
    return "\n".join(lines)


def _normalize_intent(primary_goal: str) -> str:
    """Map free-form primary_goal text → 'local' | 'global' | 'balanced'."""
    g = (primary_goal or "").lower().strip()
    # Group / balanced goals first — their descriptive prompts may mention
    # several metric names without making any single metric the sole target.
    if g.startswith("balance"):
        return "balanced"
    if g.startswith("local structure") or g.startswith("local "):
        return "local"
    if g.startswith("global structure") or g.startswith("global "):
        return "global"
    if g == "local":
        return "local"
    if g == "global":
        return "global"
    # Explicit single-metric goals route to the closest recipe family.
    if "trustworthiness" in g or "continuity" in g or "lof" in g:
        return "local"              # neighborhood-preservation metrics
    if "stress" in g or "spearman" in g:
        return "global"             # distance / rank preservation metrics
    if "silhouette" in g:
        return "local"              # cluster separation benefits from t-SNE-style recipes
    if "local" in g:
        return "local"
    if "global" in g or "topology" in g or "skeleton" in g:
        return "global"
    return "balanced"

PROMPT_TEMPLATE = """You are the Cost Function Composer. Design the objective function that the embedding will optimize — another key source of algorithmic novelty.

CONTEXT:
- Design goal: {primary_goal}
    (valid values:
      'balance both' | 'preserve local structure' | 'reveal global topology'
      OR an explicit single-metric goal such as:
        'maximum trustworthiness — …', 'maximum continuity — …',
        'maximum spearman — …', 'maximum silhouette — …',
        'maximum LOF preservation — …', 'minimum stress — …')
    If the goal text explicitly names a target metric, that metric is the SOLE
    optimization target. Every composition choice must be justified by how it
    raises/lowers that metric; all other metrics are secondary and may be traded off.
- Affinity design: {affinity_design}
- Difficulty regions: {difficulty_classification}
- Available cost primitives: {cost_primitives}

BASELINE CALIBRATION (Phase 0 — INFORMATIONAL, not prescriptive):
{baseline_section}

The baseline block tells you which family of objective the data has historically
rewarded and the metric values your novel cost design needs to clear. Treat it as
evidence, not as a recipe to copy. You are free to choose any composition pattern
that fits the design intent and the affinity Agent 4 picked.

WEAKNESS-TARGETED AUGMENTATION (if Pattern B / E / hybrid is chosen): any auxiliary
term should target `weakness_analysis.winner_weakest_metric` from the baseline block
rather than the primary metric the data already scores well on — auxiliary terms aimed
at the already-strong primary metric waste gradient capacity. Weakness→primitive map:
    - Spearman weak   → add `spectral` loss as blend_term (preserves global ranks via
                        Laplacian eigenstructure). Weight λ_global ∈ [0.05, 0.2].
    - Stress weak     → add `force_directed` or `spectral` blend_term
                        targeting distance preservation. Weight λ_global ∈ [0.05, 0.2].
    - Continuity weak → keep Pattern A; fix in Agent 4 by raising affinity k, not here.
    - Trustworthiness weak → keep Pattern A; the recipe is already optimal for T@k —
                        no cost term fixes low T more effectively than a cleaner affinity.
  The augmentation term lives in COST, never in AFFINITY (affinity blends dilute T@k).

METRICS ENGINE CATALOG (classified quantitative metrics available for cost-term derivation):
{metrics_catalog}

Each metric belongs to a *bucket* that corresponds to a distinct facet of embedding
quality. You may turn any metric (or a differentiable surrogate of it) into a cost
term and choose whether it acts as the *primary objective* or as a *constraint*.

COMPOSITION PATTERNS (Reference):

Pattern A: Single-Objective (pure, unconstrained)
  Cost = Cost_primary(Y)
  No constraint terms, no auxiliary blends. The optimizer spends all its capacity
  on the primary loss.
  Use when: the HD affinity + LD kernel pair is already a well-matched "recipe" that
  targets the primary goal on its own (e.g., perplexity_gaussian + student_t(1) + KL
  for local, or a spectral objective for global). Adding auxiliary terms here
  introduces gradient noise that fights the primary objective.

Pattern B: Multi-Scale Cost (additive blend)
  Cost = λ_local · Cost_local + λ_global · Cost_global
  Example: KL divergence + Spectral loss
    Cost = 0.8 · KL(P||Q) + 0.2 · Tr(L Y^T Y)
    → Preserves local structure while maintaining global layout
  Use when: scales of both terms are comparable and user intent is 'balance both'.

Pattern C: Adaptive Weighting
  λ_i = f(local properties)
  Example: Density-adaptive local weight
    λ_local(x_i) = 1.0 - normalize(density(x_i))
    → Stronger local preservation in sparse regions

Pattern E: Constrained Multi-Objective Optimization (NEW)
  minimize    Cost_primary(Y)
  subject to  Cost_secondary_i(Y) ≤ τ_i      (or ≥ for "higher-is-better" metrics)
                                             (or τ_low ≤ Cost_secondary_i ≤ τ_high)

  Key idea: rather than blending objectives with a fixed weight, *prioritize* one
  objective and only require the others to stay "good enough". Avoids weight-tuning
  ambiguity when objectives live on incomparable scales.

  User-intent → role assignment:
    - 'preserve local structure'  → PREFER Pattern A if the HD/LD pair is a recognized
                                    local recipe (see RECIPE CHECK below). Otherwise
                                    use Pattern E with primary = local-bucket loss,
                                    constraints on global-bucket metrics.
    - 'reveal global topology'    → primary = global-bucket loss,
                                    constrain local-bucket metrics within range
    - 'balance both'              → use Pattern B blend, OR symmetric Pattern E
                                    with two-sided constraints around target values

  RECIPE CHECK (applies only when primary_goal contains "local"):
    If Agent 4's affinity_design already specifies one of the canonical local recipes —
      (a) primary affinity = perplexity_gaussian  AND  LD kernel = student_t (dof≈1)
          AND  cost = kl_divergence      [t-SNE recipe]
      (b) primary affinity = umap_fuzzy  AND  LD kernel = umap_ab(a, b)
          AND  cost = cross_entropy      [UMAP recipe]
    — then choose Pattern A (no constraint terms, no stress blends). The paired kernel
    already handles global crowding; adding a stress / Spearman constraint term
    introduces noisy stochastic gradients that degrade Trustworthiness@k in practice
    (observed regression: T@10 dropped 0.98 → 0.94 when a kruskal-stress penalty was
    added to a t-SNE recipe).
    Only escalate to Pattern E if the user_budget in design_constraints explicitly
    demands a hard global-metric floor that the recipe alone cannot meet.

  Threshold τ sources:
    - baseline_relative: e.g., "trustworthiness must be ≥ 0.9 × UMAP baseline"
    - absolute_floor:    e.g., "Spearman ρ must be ≥ 0.6"
    - user_budget:       value supplied by the user via design_constraints

  Enforcement options (Optimization Strategy Designer chooses based on smoothness):
    - lagrangian / augmented_lagrangian   (smooth, differentiable constraints)
    - projected_gradient                  (constraint set has cheap projection)
    - penalty                             (large λ on max(0, g(Y) - τ)²)

  Example (global preference, do not destroy local structure):
    minimize    Stress(Y) + 0.3 · (1 - Spearman(Y))
    subject to  Trustworthiness@10(Y) ≥ 0.85
               Continuity@10(Y)      ≥ 0.85

  Use when: user intent is asymmetric, scales are incomparable, or explainability
  ("we optimize X subject to Y staying healthy") matters more than a weight blend.

COST PRIMITIVES REFERENCE:
- kl_divergence: t-SNE style KL(P||Q)
  - Preserves: Local neighborhoods
  - Formula: KL = Σ p_ij log(p_ij / q_ij)
  - Good for: Local structure focus
  - Limitations: May lose global structure, crowding problem
  - Used in: t-SNE

- cross_entropy: UMAP-style cross-entropy with negative sampling
  - Preserves: Both local and some global (tunable)
  - Formula: CE = -Σ [p_ij log(q_ij) + (1-p_ij) log(1-q_ij)]
  - Good for: Balanced local/global, scalable
  - Limitations: Requires negative sampling, hyperparameter sensitive
  - Used in: UMAP, LargeVis

- spectral: Spectral embedding loss (Laplacian eigenmaps style)
  - Preserves: Global structure, manifold geometry
  - Formula: Tr(Y^T L Y) where L is graph Laplacian
  - Good for: Global structure, manifold unfolding
  - Limitations: Less emphasis on local details
  - Used in: Spectral embedding, diffusion maps

- force_directed: Force-directed layout (spring-electrical model)
  - Preserves: Interpretable neighbor relationships
  - Formula: Σ (||y_i - y_j|| - d_ij)^2
  - Good for: Visualization, interpretability
  - Limitations: Can get stuck in local minima
  - Used in: ForceAtlas2, some graph layouts

- mutual_information: Information-maximization
  - Preserves: Statistical dependencies
  - Formula: I(X; Y) = H(X) - H(X|Y)
  - Good for: Preserving statistical relationships
  - Limitations: Computationally expensive
  - Used in: Some info-theoretic methods

LD-KERNEL PRIMITIVES REFERENCE:

The cost is computed between HD affinities P and LD affinities Q. Q is determined by the
LD kernel you choose here. The HD/LD kernel pair determines whether the algorithm suffers
the crowding problem. The LD kernel is a FIRST-CLASS design choice, not an implicit detail.

- student_t(dof)
    q_ij ∝ (1 + ||y_i - y_j||^2 / dof)^(-(dof+1)/2)
    dof = 1  reproduces the Cauchy / t-SNE kernel (heavy tails → resolves crowding).
    dof → ∞  reduces to Gaussian.
    PAIR RECOMMENDATION: HD = perplexity_gaussian, cost = kl_divergence, LD = student_t(1)
                         is the t-SNE recipe. Use whenever primary_goal contains "local".

- gaussian_ld
    q_ij ∝ exp(-||y_i - y_j||^2)
    Light tails → crowding problem (medium-range neighbours compressed toward origin).
    Only use if you explicitly want pure global-scale preservation paired with a
    spectral / stress objective. Do NOT pair with kl_divergence for primary_goal = "local".

- umap_ab(a, b)
    q_ij = 1 / (1 + a * ||y_i - y_j||^(2b))
    Fit-to-min-dist family. HD = umap_fuzzy + cost = cross_entropy + LD = umap_ab
    reproduces the UMAP recipe. Good for primary_goal = "local" or "balanced".

CROWDING CHECK:
Any design that uses a light-tailed LD kernel (gaussian_ld) with primary_goal = "local"
must justify why it will not suffer crowding. Otherwise pick student_t or umap_ab.

TASKS:
1. Read the Metrics Engine catalog and bucket each metric (local / global / cluster / outlier)
2. Translate {primary_goal} → composition pattern + role assignment:
     - 'preserve local structure'  → Pattern A if RECIPE CHECK passes (t-SNE or UMAP
                                     recipe already present); otherwise Pattern E with
                                     primary = local, constraints on global
     - 'reveal global topology'    → Pattern E with primary = global, constraints on local
     - 'balance both'              → Pattern B blend OR symmetric Pattern E
3. Select primary cost primitive(s) AND the paired LD kernel. For primary_goal = "local",
   the HD/LD pair must be one of:
     - perplexity_gaussian + student_t(dof=1) + kl_divergence   [t-SNE recipe]
     - umap_fuzzy         + umap_ab(a, b)    + cross_entropy    [UMAP recipe]
     - any composition where HD is a smooth density-adaptive primitive AND LD is heavy-tailed
   (For Pattern E) add any constraint terms.
4. For each constraint term, specify τ value and threshold_source (baseline_relative |
   absolute_floor | user_budget)
5. Specify local vs global trade-offs (λ values for Pattern B, τ values for Pattern E)
6. Define any adaptive weighting schemes

REQUIRED OUTPUT (JSON):
{{
  "cost_design": {{
    "composition_pattern": "A_pure | B_blend | C_adaptive | E_constrained | hybrid",
    "user_intent": "balanced | global | local",
    "metrics_engine_mapping": {{
      "local_bucket_metrics":   ["string (metric names from catalog)"],
      "global_bucket_metrics":  ["string"],
      "cluster_bucket_metrics": ["string"],
      "outlier_bucket_metrics": ["string"]
    }},
    "primary_objective": "string (primitive name; for Pattern E this is the primary loss)",
    "is_multiobjective": boolean,
    "ld_kernel": {{
      "primitive": "student_t | gaussian_ld | umap_ab",
      "parameters": {{ "dof": 1 }}
      /* or {{"a": 1.577, "b": 0.895}} for umap_ab */
    }},
    "components": [
      {{
        "primitive": "string (e.g., 'kl_divergence', 'spectral')",
        "role": "primary | blend_term",
        "weight": "float or 'adaptive'",
        "preserves": "string (e.g., 'local_neighborhoods', 'global_manifold_structure')",
        "formula": "string (mathematical expression if applicable)"
      }}
    ],
    "constraint_terms": [
      {{
        "primitive": "string (e.g., 'trustworthiness', 'spearman')",
        "metric_bucket": "local | global | cluster | outlier",
        "constraint": "<= τ | >= τ | in [τ_low, τ_high]",
        "threshold_value": "float (or [low, high])",
        "threshold_source": "baseline_relative | absolute_floor | user_budget",
        "enforcement": "lagrangian | augmented_lagrangian | projected_gradient | penalty",
        "rationale": "string"
      }}
    ],
    "adaptive_weighting": {{
      "enabled": boolean,
      "weight_function": "string (e.g., 'density_based', 'difficulty_based')",
      "adapts_to": ["density", "curvature", "difficulty_region"]
    }},
    "normalization": "string (e.g., 'row_normalize', 'symmetric', 'none')"
  }},
  "expected_behavior": {{
    "local_vs_global": "string (e.g., '70% local, 30% global')",
    "crowding_handling": "string (e.g., 'early_exaggeration', 'repulsive_terms')",
    "outlier_sensitivity": "low | medium | high",
    "convergence_speed": "fast | medium | slow"
  }},
  "optimization_challenges": [
    "string (e.g., 'non-convex', 'local_minima', 'gradient_instability')"
  ],
  "design_rationale": "string (2-3 sentences explaining why this design fits the goal)"
}}"""


class CostComposerAgent(BaseAgent):
    """Designs the cost/objective function for novel DR algorithms."""

    AGENT_NAME = "CostComposer"
    AGENT_ID = 5

    def build_prompt(self, context: dict) -> str:
        design_constraints = context.get("design_constraints", {})
        agent2_output = context.get("agent2_output", {})
        agent4_output = context.get("agent4_output", {})
        available_methods = context.get("available_methods", {})

        primary_goal = design_constraints.get("primary_goal", "balance both")
        difficulty_classification = agent2_output.get("difficulty_classification", {})
        affinity_design = agent4_output.get("affinity_design", {})

        disabled = set(context.get("disabled_components", []))

        # Get cost primitives from context or use defaults
        cost_primitives = available_methods.get("cost_primitives", [
            "kl_divergence", "cross_entropy", "spectral", "force_directed", "mutual_information"
        ])
        cost_primitives_str = (
            "(none provided — design the objective from first principles)"
            if "primitive_library" in disabled else str(cost_primitives)
        )

        # Allow callers (e.g., planner) to override the metrics catalog
        metrics_catalog = context.get("metrics_catalog", METRICS_ENGINE_CATALOG)
        catalog_str = _format_metrics_catalog(metrics_catalog)

        intent = _normalize_intent(primary_goal)
        primary_goal_annotated = f"{primary_goal}  (normalized intent: {intent})"

        from .agent_4_affinity_designer import _format_baseline_section, _strip_between
        baseline_section = _format_baseline_section(context.get("baseline_profile"))

        prompt = PROMPT_TEMPLATE.format(
            primary_goal=primary_goal_annotated,
            affinity_design=str(affinity_design),
            difficulty_classification=str(difficulty_classification),
            cost_primitives=cost_primitives_str,
            metrics_catalog=catalog_str,
            baseline_section=baseline_section,
        )
        # Ablation: strip the composition-pattern / primitive-library prompt blocks.
        if "patterns" in disabled:
            prompt = _strip_between(prompt, "COMPOSITION PATTERNS (Reference):",
                                    "COST PRIMITIVES REFERENCE:",
                                    "[composition-pattern guidance removed for ablation]")
        if "primitive_library" in disabled:
            prompt = _strip_between(prompt, "COST PRIMITIVES REFERENCE:", "CROWDING CHECK:",
                                    "[cost/LD-kernel primitive library removed for ablation]")
        return prompt

    def system_message(self) -> str:
        return (
            "You are a specialized agent that designs cost/objective functions "
            "for novel dimensionality reduction algorithms. You select and compose "
            "cost function primitives, specify local/global trade-offs, and define "
            "adaptive weighting schemes. Respond with valid JSON only."
        )

    def _validate(self, output: dict) -> None:
        """Validate that required fields are present."""
        required_sections = ["cost_design", "expected_behavior", "optimization_challenges"]
        for section in required_sections:
            if section not in output:
                self.logger.warning(f"[{self.AGENT_NAME}] Missing required section: {section}")

        # Validate cost_design structure
        if "cost_design" in output:
            cost = output["cost_design"]
            required_keys = ["primary_objective", "is_multiobjective", "components"]
            for key in required_keys:
                if key not in cost:
                    self.logger.warning(f"[{self.AGENT_NAME}] Missing cost_design key: {key}")

            if "ld_kernel" not in cost:
                self.logger.warning(
                    f"[{self.AGENT_NAME}] ld_kernel missing — crowding behaviour undefined"
                )

            # Pattern E (constrained multi-objective) requires constraint_terms
            pattern = cost.get("composition_pattern", "")
            if "E" in pattern or "constrained" in pattern.lower():
                if not cost.get("constraint_terms"):
                    self.logger.warning(
                        f"[{self.AGENT_NAME}] composition_pattern={pattern!r} "
                        "but constraint_terms is empty — Pattern E requires at least one constraint."
                    )

            # Pattern A (pure single-objective) must NOT carry constraint terms,
            # and should have exactly one primary component.
            if pattern.startswith("A") or "pure" in pattern.lower():
                if cost.get("constraint_terms"):
                    self.logger.warning(
                        f"[{self.AGENT_NAME}] composition_pattern={pattern!r} "
                        "but constraint_terms is non-empty — Pattern A must be unconstrained; "
                        "either drop constraints or switch to Pattern E."
                    )
                comps = cost.get("components", [])
                non_primary = [c for c in comps if c.get("role") != "primary"]
                if non_primary:
                    self.logger.warning(
                        f"[{self.AGENT_NAME}] Pattern A has {len(non_primary)} non-primary "
                        "component(s) — Pattern A should be single-objective (one primary only)."
                    )
