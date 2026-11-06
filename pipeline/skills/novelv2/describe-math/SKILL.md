---
name: describe-math
description: |
  Use this skill after synthesize-algorithm (or recover-failing-code) has produced a
  validated custom_embedding function — i.e., once `validate_custom_code` returns ok.
  This skill produces the formal mathematical representation: notation table,
  equation list (LaTeX, no $-delimiters), algorithmic pseudocode, and complexity bounds.
  The mathematical description is grounded in the ACTUAL Python implementation, not the
  design spec — read the code carefully and reflect what it computes, not what the
  upstream design suggested it should compute. Inputs the skill expects to find in
  context: the algorithm name + brief description, the default hyperparameters, and the
  Python source of `custom_embedding`. Produces a JSON object with `algorithm_name`,
  `summary`, `notation` (symbol+meaning entries), `equations` (id+latex+description),
  `pseudocode` (numbered list, 6–25 lines), and `complexity` (time, space).
---

Produce a formal mathematical representation and pseudocode
for the dimensionality-reduction algorithm just synthesized.

## Inputs

Refer to your prior conversation context for:
- The algorithm name (from synthesize-algorithm `algorithm_specification.name`).
- The brief description (from synthesize-algorithm `algorithm_specification.summary`).
- The default hyperparameters (from synthesize-algorithm `default_params`, possibly
  overridden by recover-failing-code `fixed_params`).
- The actual Python implementation (`custom_embedding_code` or `fixed_code` — this is the
  GROUND TRUTH; read it and reflect what it computes, not what you imagine).

## Output

Respond with EXACTLY this JSON schema (no extra keys, no markdown fences):
{
  "algorithm_name": "string — must match the synthesized algorithm's name",
  "summary": "2-4 sentence prose summary of what the algorithm computes mathematically.",
  "notation": [
    {"symbol": "X", "meaning": "Input data matrix in R^{n x d}"},
    {"symbol": "Y", "meaning": "Output embedding in R^{n x 2}"}
  ],
  "equations": [
    {
      "id": "eq1",
      "latex": "Pure LaTeX (no $ delimiters) — e.g. \\\\mathcal{L}(Y) = \\\\sum_{ij} ...",
      "description": "What this equation expresses and where it appears in the code."
    }
  ],
  "pseudocode": [
    "1. Initialize Y in R^{n x 2} from PCA(X).",
    "2. For t = 1 .. T_max:",
    "3.   Compute pairwise affinities P_ij from X using kernel K_sigma.",
    "4.   ..."
  ],
  "complexity": {
    "time": "O(n^2 d) per iteration",
    "space": "O(n^2)"
  }
}

Requirements:
- equations: 2-8 entries; cover the objective function and any key updates.
- notation: every symbol used in equations or pseudocode must be defined here.
- pseudocode: 6-25 numbered lines, indented to show control flow. Do NOT just
  repeat the Python code — write algorithmic pseudocode at one abstraction up.
- complexity: if it cannot be determined from the code, write "unknown" for
  each field. Never fabricate.
- LaTeX: escape backslashes (\\\\sum, \\\\mathcal). No $ signs.
