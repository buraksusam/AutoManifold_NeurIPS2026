"""
Worker Agent 10 — Code Error Recovery Agent

When LLM-generated custom_embedding code fails at runtime (e.g. wrong
neighbor count, shape mismatch, missing import, numerical instability),
this agent receives the failing code, the full error message + traceback,
and the dataset metadata.  It diagnoses the root cause, patches the code,
and returns a corrected version that the pipeline can re-validate
immediately.
"""

import json
from ..base_agent import BaseAgent


PROMPT_TEMPLATE = """You are the **Code Error Recovery Agent**.  A custom dimensionality-reduction
function produced by an earlier agent has crashed at runtime.  Your job is to:

1. Read the error message and traceback carefully.
2. Diagnose the root cause.
3. Produce a **corrected** version of the function that avoids the error.
4. Optionally adjust the default parameters so the code will succeed on this dataset.

═══════════════════════════════════════════════════════════
FAILING CODE
═══════════════════════════════════════════════════════════
```python
{failing_code}
```

═══════════════════════════════════════════════════════════
ERROR MESSAGE / TRACEBACK
═══════════════════════════════════════════════════════════
{error_message}

═══════════════════════════════════════════════════════════
DATASET METADATA
═══════════════════════════════════════════════════════════
- n_samples  : {n_samples}
- n_features : {n_features}
- n_classes  : {n_classes}
- has_labels : {has_labels}

═══════════════════════════════════════════════════════════
CURRENT HYPERPARAMETERS
═══════════════════════════════════════════════════════════
{current_params_json}

═══════════════════════════════════════════════════════════
PREVIOUS RECOVERY ATTEMPTS (if any)
═══════════════════════════════════════════════════════════
{previous_attempts}

═══════════════════════════════════════════════════════════
COMMON FIXES — keep these in mind
═══════════════════════════════════════════════════════════
• **Neighbor / k overflow**: n_neighbors, k, or perplexity must be < n_samples.
  Safe guard: `k = min(k, n_samples - 1)`.
  For UMAP: `n_neighbors = min(n_neighbors, n_samples - 1)`.
  For t-SNE perplexity: `perplexity = min(perplexity, (n_samples - 1) // 3)`.
  For kneighbors_graph / NearestNeighbors: clamp n_neighbors.
  For precomputed graphs: if you build a k-NN graph with k₁ but later an
  algorithm requests k₂ > k₁ neighbors, you must build the graph with
  max(k₁, k₂) neighbors from the start.

• **sklearn TSNE**: use `max_iter=` NOT `n_iter=` (renamed in sklearn ≥ 1.4).

• **Shape mismatch**: ensure the function returns an (n_samples, 2) array.

• **Import errors**: all imports MUST be inside the function body.

• **NaN / Inf**: add `np.nan_to_num(result)` or fall-back to PCA if the
  optimizer diverges.

• **Memory / timeout**: reduce internal iterations, subsample, or use
  approximate neighbors (e.g. `pynndescent` or `annoy`).

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════
Return a JSON object with these fields:

{{
  "diagnosis": "1-3 sentence explanation of the root cause",
  "fix_summary": "1-3 sentence description of the fix you applied",
  "fixed_code": "COMPLETE corrected Python function as a string (same signature: def custom_embedding(data, labels, params): ...)",
  "fixed_params": {{
    "param_name": "corrected_default_value"
  }},
  "confidence": 0.0-1.0
}}

RULES:
  1. The "fixed_code" MUST keep the exact signature: `def custom_embedding(data, labels, params):`
  2. The function MUST return a numpy array of shape (n_samples, 2).
  3. Put ALL imports inside the function body.
  4. Make the MINIMAL change needed to fix the crash — don't rewrite the whole algorithm.
  5. Add defensive guards (clamp neighbors, check shapes) so the same class of error cannot recur.
  6. "fixed_params" should include the full param dict (not just changed keys).
"""


class ErrorRecoveryAgent(BaseAgent):
    AGENT_NAME = "ErrorRecovery"
    AGENT_ID = 10

    def system_message(self) -> str:
        return (
            "You are a senior Python debugging specialist for scientific computing. "
            "You receive failing code and its traceback, then return corrected code "
            "as valid JSON only. No markdown fences, no commentary outside the JSON."
        )

    def build_prompt(self, context: dict) -> str:
        previous = context.get("previous_attempts", [])
        if previous:
            prev_text = "\n".join(
                f"Attempt {i+1}: {a.get('diagnosis','?')} → still failed: {a.get('error','?')}"
                for i, a in enumerate(previous)
            )
        else:
            prev_text = "(none)"

        params = context.get("current_params", {})

        return PROMPT_TEMPLATE.format(
            failing_code=context.get("failing_code", "(not provided)"),
            error_message=context.get("error_message", "(not provided)"),
            n_samples=context.get("n_samples", "?"),
            n_features=context.get("n_features", "?"),
            n_classes=context.get("n_classes", "?"),
            has_labels=context.get("has_labels", "?"),
            current_params_json=json.dumps(params, indent=2, default=str),
            previous_attempts=prev_text,
        )
