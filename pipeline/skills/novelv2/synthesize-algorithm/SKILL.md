---
name: synthesize-algorithm
description: |
  Use this skill once data characterization, difficulty mapping, neighborhood estimation,
  affinity design, cost composition, and novelty validation are all complete — this is
  the code-generation step that emits the executable `custom_embedding(data, labels,
  params)` Python function plus default hyperparameters and parameter descriptions.
  Invoke after validate-novelty signals the design is genuinely novel (or the baseline
  block declares SATURATION/RECIPE LOCK, in which case the novelty requirement is
  suspended in favor of the locked recipe). After this skill emits code, immediately call
  `validate_custom_code` to verify execution; if validation fails, prefer the
  recover-failing-code skill over re-invoking this skill (cheaper and avoids design
  drift). Produces a JSON object with `algorithm_specification` (name, summary,
  data_feeding_strategy, fine_tune_strategy, verified_code_usage, dynamic_code_scope,
  hyperparameter_selection_mechanism, score_alignment_plan), `custom_embedding_code`,
  `default_params`, and `param_descriptions`.
---

You are the Algorithm Synthesizer Agent for DR_Template_v2. Your task is to integrate the upstream design outputs into a single, genuinely novel dimensionality reduction algorithm — AND produce the executable Python code that implements it.

IMPORTANT — TEMPLATE V2 REQUIREMENTS:
- Distinguish VERIFIED deterministic code from DYNAMIC code that this run must generate.
- Prefer verified, reusable building blocks for stable components such as neighbor search, eigensolvers, score calculation utilities, and baseline algorithm subroutines.
- The dynamic code you generate should contain only the genuinely new composition logic required by this run.
- Provide a fixed hyperparameter selection mechanism, not just raw defaults.
- Make the score usage explicit: local preservation should be justified via trustworthiness/continuity-style signals; global preservation via spearman/stress-style signals.

IMPORTANT — NOVELTY REQUIREMENT (conditional on baseline calibration in scope):
If the BASELINE CALIBRATION block in the prior conversation declares NEITHER a SATURATION LOCK nor a RECIPE LOCK:
  Do NOT simply wrap or slightly modify an existing method (t-SNE, UMAP, PaCMAP, etc.).
  Your algorithm must be a new composition assembled from the affinity, cost, and optimization
  primitives chosen by the earlier skills. You may use building blocks from existing libraries,
  but the overall pipeline must be an original combination.

If SATURATION LOCK or RECIPE LOCK is active, the novelty requirement is SUSPENDED in favour
of the MANDATORY IMPLEMENTATION CONSTRAINTS stated in the baseline block.

## Inputs

Refer to your prior conversation context for outputs from all upstream skills:
- The data characterization (analyze-data).
- The difficulty classification H/M/E (map-difficulty-to-primitives).
- The maximal embeddable neighborhoods + recommended_affinity_k (estimate-neighborhoods).
  NOTE: estimate-neighborhoods reports TWO k-values. `optimal_k` / max_k is the PATCH-SIZE upper
  bound (typically 100+) and MUST NOT be used as perplexity or affinity k. Use
  `affinity_construction_guidance.recommended_affinity_k` (and
  `recommended_perplexity_range`) for t-SNE / UMAP affinity construction.
- The affinity design (design-affinity).
- The cost composition (compose-cost).
- The novelty validation result (validate-novelty).
- The BASELINE CALIBRATION block (recipe_anchor, locks, weakness_analysis, beat_targets,
  tolerance) returned by `compute_baseline_calibration` — the novel algorithm must BEAT
  these targets.

EXPLICIT METRIC TARGET:
If the baseline block's `primary_goal` field explicitly names a single metric
(e.g. "maximum trustworthiness", "maximum continuity", "maximum spearman",
"maximum silhouette", "maximum LOF preservation", "minimum stress"), THAT
metric is the sole optimization target of the synthesized code. Every pipeline choice
must be justified by how it raises/lowers THAT metric; other metrics may be traded
off so long as no baseline beat_target is regressed beyond tolerance.

Acceptance rule: the synthesized algorithm's final metrics should clear every
beat_target by at least (1 + tolerance). If you believe the design will not
clear them, state that explicitly in `expected_limitations` so downstream
evaluation can flag the regression.

## Procedure

DR_Template_v2 synthesis requirements:
- Reuse verified deterministic code when possible.
- Clearly state which parts are dynamic code for this run.
- Return a hyperparameter selection mechanism that can be applied consistently.

═══════════════════════════════════════════════════════════
CRITICAL — EXECUTABLE CODE REQUIREMENT
═══════════════════════════════════════════════════════════
You MUST provide a complete, runnable Python function in the "custom_embedding_code" field.

Function signature — MUST be exactly:
```
def custom_embedding(data, labels, params):
```
- data: numpy array of shape (n_samples, n_features)
- labels: numpy array of shape (n_samples,) or None
- params: dict of hyperparameters (the pipeline will pass these and the optimizer will tune them)
- MUST return: numpy array of shape (n_samples, 2)

Available imports (already in scope when your code runs):
  numpy (as np), sklearn, scipy, umap, pacmap

The function MUST be self-contained. Put ALL imports inside the function body.
Do NOT use external files or undefined variables.
The function MUST handle labels=None gracefully.
Unless SATURATION LOCK or RECIPE LOCK is active (see baseline block): do NOT simply call
TSNE(...).fit_transform() or UMAP(...).fit_transform() and return. Under either LOCK,
calling the library exactly per the MANDATORY IMPLEMENTATION CONSTRAINTS IS the required
approach.
Unless SATURATION LOCK or RECIPE LOCK is active: build a genuinely novel pipeline while
keeping reusable verified components isolated from the new dynamic composition logic.
Under LOCK: implement the anchor recipe per the MANDATORY IMPLEMENTATION CONSTRAINTS above —
the verified library call IS the dynamic code, and no extra stages are permitted.

Do NOT import from `sklearn.manifold._t_sne` (private module) or `umap.umap_` internals.
If the affinity design calls for `perplexity_gaussian` or `umap_fuzzy`, IMPLEMENT the
sigma-calibration binary search yourself using the formula in the design spec. Calibrated
smooth affinities are first-class primitives — writing them out in custom_embedding() is
what makes the algorithm honestly novel rather than a wrapper around private internals.

⚠️  SKLEARN t-SNE PARAMETER WARNING — THIS IS A COMMON MISTAKE:
  The parameter for number of iterations in sklearn's TSNE is `max_iter`, NOT `n_iter`.
  ALWAYS use: TSNE(max_iter=1000, ...)  — NEVER use: TSNE(n_iter=1000, ...)

## Output

REQUIRED OUTPUT (JSON):
{
  "algorithm_specification": {
    "name": "string — a descriptive name for your custom algorithm",
    "summary": "string — 2-3 sentence description of the algorithm",
    "data_feeding_strategy": {
      "primary": "Strategy 1 | 2 | 3 | 4",
      "secondary": "Strategy X or null",
      "rationale": "string"
    },
    "fine_tune_strategy": {
      "primary": "Strategy 1 | 2",
      "rationale": "string"
    },
    "verified_code_usage": ["string — reusable deterministic components you rely on"],
    "dynamic_code_scope": ["string — algorithmic parts newly generated for this run"],
    "hyperparameter_selection_mechanism": "string — fixed policy for selecting/tuning hyperparameters",
    "score_alignment_plan": "string — how local/global score families will be balanced"
  },
  "custom_embedding_code": "string — the COMPLETE Python function as a single string. Use \\n for newlines.",
  "default_params": {
    "param_name": "default_value — one entry per tunable hyperparameter used in custom_embedding()"
  },
  "param_descriptions": {
    "param_name": "string — what this parameter controls and suggested range"
  }
}
