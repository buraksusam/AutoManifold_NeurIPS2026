"""
Worker Agent 8 — Algorithm Synthesizer Agent

DR_Template Sections 2.1, 2.2, 2.3, and Section 3.
Integrates all component outputs into a complete DR algorithm specification
**and produces executable Python code** that the pipeline runs directly.
"""

import json
from ..base_agent import BaseAgent

PROMPT_TEMPLATE = """You are the Algorithm Synthesizer Agent. Your task is to integrate all component designs from the other agents into a single, **genuinely novel** dimensionality reduction algorithm — AND produce the executable Python code that implements it.

IMPORTANT — NOVELTY REQUIREMENT:
Do NOT simply wrap or slightly modify an existing method (t-SNE, UMAP, PaCMAP, etc.).
Your algorithm must be a *new composition* assembled from the affinity, cost, and optimization
primitives chosen by the earlier agents. You may USE building blocks from existing libraries
(e.g. kNN graphs from sklearn, spectral decomposition from scipy) but the overall pipeline
must be an original combination — not a thin wrapper around one established method.

INPUT — Outputs from all agents:
- Data characterization: {agent1_summary}
- Difficulty classification (H/M/E): {agent2_summary}
- Maximal embeddable neighborhoods: {agent3_summary}
  NOTE: Agent 3 reports TWO k-values. `optimal_k` / max_k is the PATCH-SIZE upper
  bound (typically 100+) and MUST NOT be used as perplexity or affinity k. Use
  `affinity_construction_guidance.recommended_affinity_k` (and
  `recommended_perplexity_range`) for t-SNE / UMAP affinity construction.

BASELINE CALIBRATION (Phase 0) — the novel algorithm must BEAT these:
{baseline_section}

EXPLICIT METRIC TARGET:
If the baseline block's `primary_goal` field explicitly names a single metric
(e.g. "maximum trustworthiness", "maximum continuity", "maximum spearman",
"maximum silhouette", "maximum LOF preservation", "minimum stress"), THAT
metric is the sole
optimization target of the synthesized code. Every pipeline choice must be
justified by how it raises/lowers THAT metric; other metrics may be traded
off so long as no baseline beat_target is regressed beyond tolerance.

Acceptance rule: the synthesized algorithm's final metrics should clear every
beat_target by at least (1 + tolerance). If you believe the design will not
clear them, state that explicitly in `expected_limitations` so downstream
evaluation can flag the regression.
- Patch stitching algorithm: {agent4_summary}
- Impossible-to-embed points: {agent5_summary}
- Hyperparameter optimization trajectory: {agent6_summary}
- Novelty validation: {agent12_summary}

DATA FEEDING STRATEGIES (DR_Template Section 2.1) — select one or combine:
  Strategy 1: "Prepare three different visualisation plots, each by prioritising embedding accuracy for one of the three groups (H, M, and E)."
  Strategy 2: "Divide data points into local patches based on maximal embeddable neighbourhoods. Embed each local patch, and then apply local patch stitching algorithms to combine."
  Strategy 3: "Incrementally optimise embeddings in the order of easy, moderate, and hard groups."
  Strategy 4: "Visualise impossible to embed points separately."

FINE-TUNE STRATEGIES (DR_Template Section 2.2):
  Strategy 1: "Identify some local patches with particularly low scores, and directly fine-tune embedding positions of those patches to boost the scores up."
  Strategy 2: "Select local patches to directly fine-tune their embedding positions so that the overall reliability scores have the highest chance to increase."

HYPERPARAMETER OPTIMIZATION (DR_Template Section 2.3 / Paper):
  θ*_A = arg min_{{θ ∈ H}} Σ wi * Oi(Z(A,θ)), using iterative LLM-guided pipeline with dual implicit/explicit scoring.

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
Do NOT simply call TSNE(...).fit_transform() or UMAP(...).fit_transform() and return.
Build a genuinely novel pipeline from the primitives the earlier agents selected. You may
use library utilities (kNN, eigensolvers, optimizers) as building blocks.

Do NOT import from `sklearn.manifold._t_sne` (private module) or `umap.umap_` internals.
If the affinity design calls for `perplexity_gaussian` or `umap_fuzzy`, IMPLEMENT the
sigma-calibration binary search yourself using the formula supplied in the design spec.
Calibrated smooth affinities are first-class primitives in this pipeline — writing them out
in your custom_embedding() is what makes the algorithm honestly novel rather than a thin
wrapper around a baseline's private internals.

⚠️  SKLEARN t-SNE PARAMETER WARNING — THIS IS A COMMON MISTAKE:
  The parameter for number of iterations in sklearn's TSNE is `max_iter`, NOT `n_iter`.
  `n_iter` was renamed to `max_iter` in sklearn >= 1.4 and will raise a TypeError.
  ALWAYS use: TSNE(max_iter=1000, ...)  — NEVER use: TSNE(n_iter=1000, ...)

Example — a simple passthrough would be:
```python
def custom_embedding(data, labels, params):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=params.get('perplexity', 30)).fit_transform(data)
```

But you should do MUCH more: implement patch-based embedding, difficulty-aware strategies, outlier handling, fine-tuning — whatever your algorithm specification calls for.

REQUIRED OUTPUT (JSON):
{{
  "algorithm_specification": {{
    "name": "string — a descriptive name for your custom algorithm",
    "summary": "string — 2-3 sentence description of the algorithm",
    "data_feeding_strategy": {{
      "primary": "Strategy 1 | 2 | 3 | 4",
      "secondary": "Strategy X or null",
      "rationale": "string"
    }},
    "fine_tune_strategy": {{
      "primary": "Strategy 1 | 2",
      "rationale": "string"
    }}
  }},
  "custom_embedding_code": "string — the COMPLETE Python function as a single string. Use \\n for newlines.",
  "default_params": {{
    "param_name": "default_value — one entry per tunable hyperparameter used in custom_embedding()"
  }},
  "param_descriptions": {{
    "param_name": "string — what this parameter controls and suggested range"
  }}
}}"""

PROMPT_TEMPLATE_ALL_LLM = """You are the Algorithm Code Synthesizer. Your task is to produce executable Python code that implements the algorithm specification designed by the AllLLMDesignerAgent.

ALGORITHM SPECIFICATION (designer output):
{design_spec}

DATA CHARACTERIZATION (from Agent 1):
{agent1_summary}

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
- params: dict of hyperparameters (tunable by the optimizer)
- MUST return: numpy array of shape (n_samples, 2)

Available imports (already in scope when your code runs):
  numpy (as np), sklearn, scipy, umap, pacmap

The function MUST be self-contained. Put ALL imports inside the function body.
Do NOT use external files or undefined variables.
The function MUST handle labels=None gracefully.

⚠️  SKLEARN t-SNE PARAMETER WARNING:
  Use `max_iter`, NOT `n_iter`. ALWAYS: TSNE(max_iter=1000, ...)  NEVER: TSNE(n_iter=1000, ...)

REQUIRED OUTPUT (JSON):
{{
  "algorithm_specification": {{
    "name": "string — algorithm name from the spec",
    "summary": "string — 2-3 sentence description"
  }},
  "custom_embedding_code": "string — the COMPLETE Python function as a single string. Use \\n for newlines.",
  "default_params": {{
    "param_name": default_value
  }},
  "param_descriptions": {{
    "param_name": "string — what this parameter controls and suggested range"
  }}
}}"""

PROMPT_TEMPLATE_V2_NOVEL = """You are the Algorithm Synthesizer Agent for DR_Template_v2. Your task is to integrate the upstream design outputs into a single, genuinely novel dimensionality reduction algorithm — AND produce the executable Python code that implements it.

IMPORTANT — TEMPLATE V2 REQUIREMENTS:
- Distinguish VERIFIED deterministic code from DYNAMIC code that this run must generate.
- Prefer verified, reusable building blocks for stable components such as neighbor search, eigensolvers, score calculation utilities, and baseline algorithm subroutines.
- The dynamic code you generate should contain only the genuinely new composition logic required by this run.
- Provide a fixed hyperparameter selection mechanism, not just raw defaults.
- Make the score usage explicit: local preservation should be justified via trustworthiness/continuity-style signals; global preservation via spearman/stress-style signals.

IMPORTANT — NOVELTY REQUIREMENT:
Do NOT simply wrap or slightly modify an existing method (t-SNE, UMAP, PaCMAP, etc.).
Your algorithm must be a new composition assembled from the affinity, cost, and optimization
primitives chosen by the earlier agents. You may use building blocks from existing libraries,
but the overall pipeline must be an original combination.

INPUT — Outputs from all agents:
- Data characterization: {agent1_summary}
- Difficulty classification (H/M/E): {agent2_summary}
- Maximal embeddable neighborhoods: {agent3_summary}
  NOTE: Agent 3 reports TWO k-values. `optimal_k` / max_k is the PATCH-SIZE upper
  bound (typically 100+) and MUST NOT be used as perplexity or affinity k. Use
  `affinity_construction_guidance.recommended_affinity_k` (and
  `recommended_perplexity_range`) for t-SNE / UMAP affinity construction.

BASELINE CALIBRATION (Phase 0) — the novel algorithm must BEAT these:
{baseline_section}

EXPLICIT METRIC TARGET:
If the baseline block's `primary_goal` field explicitly names a single metric
(e.g. "maximum trustworthiness", "maximum continuity", "maximum spearman",
"maximum silhouette", "maximum LOF preservation", "minimum stress"), THAT
metric is the sole
optimization target of the synthesized code. Every pipeline choice must be
justified by how it raises/lowers THAT metric; other metrics may be traded
off so long as no baseline beat_target is regressed beyond tolerance.

Acceptance rule: the synthesized algorithm's final metrics should clear every
beat_target by at least (1 + tolerance). If you believe the design will not
clear them, state that explicitly in `expected_limitations` so downstream
evaluation can flag the regression.
- Affinity graph design (Agent 4 — AffinityDesigner): {agent4_summary}
- Cost function composition (Agent 5 — CostComposer): {agent5_summary}
- Selected optimization recipe (auto-chosen from cost type + budget): {agent6_summary}
- Novelty validation verdict (Agent 12 — NoveltyValidator): {agent12_summary}
  Honour `suggested_name` if present, and reflect `theoretical_differences`
  in `algorithm_specification.summary`. `is_genuinely_novel == false` MUST
  be addressed in `dynamic_code_scope` by changing the composition, not
  the wording.

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
Do NOT simply call TSNE(...).fit_transform() or UMAP(...).fit_transform() and return.
Build a genuinely novel pipeline while keeping reusable verified components isolated from
the new dynamic composition logic.

Do NOT import from `sklearn.manifold._t_sne` (private module) or `umap.umap_` internals.
If the affinity design calls for `perplexity_gaussian` or `umap_fuzzy`, IMPLEMENT the
sigma-calibration binary search yourself using the formula in the design spec. Calibrated
smooth affinities are first-class primitives — writing them out in custom_embedding() is
what makes the algorithm honestly novel rather than a wrapper around private internals.

⚠️  SKLEARN t-SNE PARAMETER WARNING — THIS IS A COMMON MISTAKE:
  The parameter for number of iterations in sklearn's TSNE is `max_iter`, NOT `n_iter`.
  ALWAYS use: TSNE(max_iter=1000, ...)  — NEVER use: TSNE(n_iter=1000, ...)

REQUIRED OUTPUT (JSON):
{{
  "algorithm_specification": {{
    "name": "string — a descriptive name for your custom algorithm",
    "summary": "string — 2-3 sentence description of the algorithm",
    "data_feeding_strategy": {{
      "primary": "Strategy 1 | 2 | 3 | 4",
      "secondary": "Strategy X or null",
      "rationale": "string"
    }},
    "fine_tune_strategy": {{
      "primary": "Strategy 1 | 2",
      "rationale": "string"
    }},
    "verified_code_usage": ["string — reusable deterministic components you rely on"],
    "dynamic_code_scope": ["string — algorithmic parts newly generated for this run"],
    "hyperparameter_selection_mechanism": "string — fixed policy for selecting/tuning hyperparameters",
    "score_alignment_plan": "string — how local/global score families will be balanced"
  }},
  "custom_embedding_code": "string — the COMPLETE Python function as a single string. Use \\n for newlines.",
  "default_params": {{
    "param_name": "default_value — one entry per tunable hyperparameter used in custom_embedding()"
  }},
  "param_descriptions": {{
    "param_name": "string — what this parameter controls and suggested range"
  }}
}}"""

PROMPT_TEMPLATE_ALL_LLM_V2 = """You are the Algorithm Code Synthesizer for DR_Template_v2. Your task is to produce executable Python code that implements the algorithm specification designed by the AllLLMDesignerAgent.

ALGORITHM SPECIFICATION (designer output):
{design_spec}

DATA CHARACTERIZATION (from Agent 1):
{agent1_summary}

IMPORTANT — TEMPLATE V2 REQUIREMENTS:
- Respect the designer's separation between VERIFIED deterministic components and DYNAMIC components.
- Prefer stable, reusable library implementations for verified deterministic subroutines.
- Generate dynamic code only for the novel, run-specific algorithm composition.
- Preserve the designer's fixed hyperparameter-selection policy in both defaults and code structure.
- Make explicit how the implementation supports local-score and global-score evaluation.

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
- params: dict of hyperparameters (tunable by the optimizer)
- MUST return: numpy array of shape (n_samples, 2)

Available imports (already in scope when your code runs):
  numpy (as np), sklearn, scipy, umap, pacmap

The function MUST be self-contained. Put ALL imports inside the function body.
Do NOT use external files or undefined variables.
The function MUST handle labels=None gracefully.

⚠️  SKLEARN t-SNE PARAMETER WARNING:
  Use `max_iter`, NOT `n_iter`. ALWAYS: TSNE(max_iter=1000, ...)  NEVER: TSNE(n_iter=1000, ...)

REQUIRED OUTPUT (JSON):
{{
  "algorithm_specification": {{
    "name": "string — algorithm name from the spec",
    "summary": "string — 2-3 sentence description",
    "verified_code_usage": ["string — deterministic/reused subroutines honored from the design"],
    "dynamic_code_scope": ["string — newly generated algorithmic logic implemented here"],
    "hyperparameter_selection_mechanism": "string — fixed policy used to pick or tune parameters",
    "score_alignment_plan": "string — how local/global score families are handled"
  }},
  "custom_embedding_code": "string — the COMPLETE Python function as a single string. Use \\n for newlines.",
  "default_params": {{
    "param_name": default_value
  }},
  "param_descriptions": {{
    "param_name": "string — what this parameter controls and suggested range"
  }}
}}"""


def _format_baseline_for_synthesizer(profile: dict | None) -> str:
    """Compact baseline_profile block for the synthesizer prompt."""
    if not profile:
        return "  (no baseline calibration available)"
    winner = profile.get("winner", {})
    anchor = profile.get("recipe_anchor", {})
    targets = profile.get("beat_targets", {})

    family = (anchor.get("family") or "").lower()
    hd = anchor.get("hd_affinity", "?")
    ld = anchor.get("ld_kernel", {}).get("primitive", "?")
    cost = anchor.get("cost", "?")

    metric_lines = []
    for k, v in (targets or {}).items():
        if k == "tolerance":
            continue
        metric_lines.append(f"    {k} = {v}")
    metrics_block = "\n".join(metric_lines) if metric_lines else "    (none reported)"

    lines = [
        f"  Winner: {winner.get('name','?')} "
        f"(intent='{profile.get('intent','?')}', score={winner.get('primary_goal_score','?')})",
        f"  What worked on this data: {family} — hd={hd}, ld={ld}, cost={cost}",
        f"  Metrics to beat (tolerance={targets.get('tolerance')}, "
        "direction governed by primary_goal):",
        metrics_block,
        f"  Winner weakness: {profile.get('weakness_analysis',{}).get('winner_weakest_metric','?')}",
        "",
        "     → INFORMATIONAL: this tells you which structural family the data rewards "
        "and which numbers your novel pipeline needs to clear. It is NOT a recipe to "
        "wrap. Build a genuinely novel composition from the upstream agent designs; "
        "use the baseline only as evidence and as the score floor to surpass.",
    ]
    return "\n".join(lines)


class AlgorithmSynthesizerAgent(BaseAgent):
    AGENT_NAME = "AlgorithmSynthesizer"
    AGENT_ID = 8

    def __init__(self, llm_client, mode: str = "novel"):
        super().__init__(llm_client)
        self._mode = mode

    def build_prompt(self, context: dict) -> str:
        # Constructor arg is authoritative. The context "mode" key is a fallback
        # for callers that construct the agent with the default ("novel") but pass
        # mode via the shared context dict (e.g. AllLLMPlanner's _phase3 call).
        if self._mode == "all_llm_v2":
            return self._build_all_llm_v2_prompt(context)
        if self._mode == "novel_v2":
            return self._build_novel_v2_prompt(context)
        if self._mode == "all_llm" or (self._mode == "novel" and context.get("mode") in {"all_llm", "allLLMv2"}):
            return self._build_all_llm_prompt(context)
        return self._build_novel_prompt(context)

    def _build_novel_prompt(self, context: dict) -> str:
        def _summarize(key, fallback="Not available"):
            val = context.get(key)
            if val is None:
                return fallback
            if isinstance(val, dict):
                return json.dumps(val, indent=2, default=str)[:3000]
            return str(val)[:3000]

        return PROMPT_TEMPLATE.format(
            agent1_summary=_summarize("agent1_output"),
            agent2_summary=_summarize("agent2_output"),
            agent3_summary=_summarize("agent3_output"),
            agent4_summary=_summarize("agent4_output"),
            agent5_summary=_summarize("agent5_output"),
            agent6_summary=_summarize("agent6_output"),
            agent12_summary=_summarize("agent12_output"),
            baseline_section=_format_baseline_for_synthesizer(context.get("baseline_profile")),
        )

    def _build_all_llm_prompt(self, context: dict) -> str:
        design_spec = context.get("design_spec", {})
        agent1 = context.get("agent1_output", {})
        return PROMPT_TEMPLATE_ALL_LLM.format(
            design_spec=json.dumps(design_spec, indent=2, default=str)[:4000],
            agent1_summary=json.dumps(agent1, indent=2, default=str)[:2000],
        )

    def _build_novel_v2_prompt(self, context: dict) -> str:
        def _summarize(key, fallback="Not available"):
            val = context.get(key)
            if val is None:
                return fallback
            if isinstance(val, dict):
                return json.dumps(val, indent=2, default=str)[:3000]
            return str(val)[:3000]

        return PROMPT_TEMPLATE_V2_NOVEL.format(
            agent1_summary=_summarize("agent1_output"),
            agent2_summary=_summarize("agent2_output"),
            agent3_summary=_summarize("agent3_output"),
            agent4_summary=_summarize("agent4_output"),
            agent5_summary=_summarize("agent5_output"),
            agent6_summary=_summarize("agent6_output"),
            agent12_summary=_summarize("agent12_output"),
            baseline_section=_format_baseline_for_synthesizer(context.get("baseline_profile")),
        )

    def _build_all_llm_v2_prompt(self, context: dict) -> str:
        design_spec = context.get("design_spec", {})
        agent1 = context.get("agent1_output", {})
        return PROMPT_TEMPLATE_ALL_LLM_V2.format(
            design_spec=json.dumps(design_spec, indent=2, default=str)[:4000],
            agent1_summary=json.dumps(agent1, indent=2, default=str)[:2000],
        )

    def system_message(self) -> str:
        return (
            "You are a senior algorithm designer who synthesizes algorithm specifications "
            "into cohesive dimensionality reduction implementations. You produce executable "
            "Python code that faithfully implements the provided specification. "
            "Respond with valid JSON only."
        )
