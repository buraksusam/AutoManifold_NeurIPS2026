"""
MathPseudocodeAgent — produces formal mathematical representation + pseudocode
for an LLM-generated DR algorithm.

This agent runs *after* the final executable code is settled (post-implementation,
post-recovery). It reads the actual code, the algorithm name/description, and the
default parameters, then asks the LLM to produce:

  - A list of LaTeX equations describing the algorithm's objective + key steps
  - A symbol/notation table
  - A numbered pseudocode listing
  - A short narrative summary

Used by all three planner modes: ablation, novel/novelv2, all_llm.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


_SYSTEM_MESSAGE = (
    "You are a mathematical expositor for a dimensionality-reduction research "
    "pipeline. Given an algorithm's name, description, and concrete Python "
    "implementation, you produce its formal mathematical representation and "
    "rigorous pseudocode. You always reflect what the code ACTUALLY does — never "
    "invent steps that are not present in the code, and never omit steps that "
    "are. Output strictly valid JSON with no markdown fences and no commentary."
)


_PROMPT_TEMPLATE = """Produce a formal mathematical representation and pseudocode
for the dimensionality-reduction algorithm described below.

ALGORITHM NAME:
{name}

BRIEF DESCRIPTION:
{description}

DEFAULT HYPERPARAMETERS:
{params_json}

ACTUAL PYTHON IMPLEMENTATION (this is the ground truth — read it and reflect
what it computes, not what you imagine):
```python
{code}
```

Respond with EXACTLY this JSON schema (no extra keys, no markdown fences):
{{
  "algorithm_name": "{name}",
  "summary": "2-4 sentence prose summary of what the algorithm computes mathematically.",
  "notation": [
    {{"symbol": "X", "meaning": "Input data matrix in R^{{n x d}}"}},
    {{"symbol": "Y", "meaning": "Output embedding in R^{{n x 2}}"}}
  ],
  "equations": [
    {{
      "id": "eq1",
      "latex": "Pure LaTeX (no $ delimiters) — e.g. \\\\mathcal{{L}}(Y) = \\\\sum_{{ij}} ...",
      "description": "What this equation expresses and where it appears in the code."
    }}
  ],
  "pseudocode": [
    "1. Initialize Y in R^{{n x 2}} from PCA(X).",
    "2. For t = 1 .. T_max:",
    "3.   Compute pairwise affinities P_ij from X using kernel K_sigma.",
    "4.   ..."
  ],
  "complexity": {{
    "time": "O(n^2 d) per iteration",
    "space": "O(n^2)"
  }}
}}

Requirements:
- equations: 2-8 entries; cover the objective function and any key updates.
- notation: every symbol used in equations or pseudocode must be defined here.
- pseudocode: 6-25 numbered lines, indented to show control flow. Do NOT just
  repeat the Python code — write algorithmic pseudocode at one abstraction up.
- complexity: if it cannot be determined from the code, write "unknown" for
  each field. Never fabricate.
- LaTeX: escape backslashes (\\\\sum, \\\\mathcal). No $ signs.
"""


class MathPseudocodeAgent(BaseAgent):
    AGENT_NAME = "MathPseudocode"
    AGENT_ID = 11

    def system_message(self) -> str:
        return _SYSTEM_MESSAGE

    def build_prompt(self, context: dict) -> str:
        import json as _json
        name = context.get("algorithm_name") or "Unnamed Algorithm"
        description = context.get("description") or ""
        code = context.get("code") or ""
        params = context.get("default_params") or {}
        # Trim very long code so the prompt stays reasonable; keep head+tail.
        if len(code) > 8000:
            head = code[:5000]
            tail = code[-2500:]
            code = head + "\n# ... (truncated) ...\n" + tail
        params_json = _json.dumps(params, indent=2, default=str)
        if len(params_json) > 1500:
            params_json = params_json[:1500] + " ... (truncated)"
        return _PROMPT_TEMPLATE.format(
            name=name,
            description=description or "(no description provided)",
            params_json=params_json,
            code=code,
        )

    def _validate(self, output: dict) -> None:
        # Soft validation — don't raise; just log gaps so we still save the result.
        for key in ("equations", "pseudocode", "notation"):
            if key not in output:
                self.logger.warning(
                    f"[{self.AGENT_NAME}] missing key '{key}' in output"
                )

    # ------------------------------------------------------------------
    # Markdown rendering helper (used by the planner after run())
    # ------------------------------------------------------------------

    @staticmethod
    def render_markdown(spec: Dict[str, Any]) -> str:
        name = spec.get("algorithm_name", "Algorithm")
        lines: List[str] = [f"# {name} — Mathematical Representation & Pseudocode", ""]

        summary = spec.get("summary")
        if summary:
            lines += ["## Summary", "", str(summary), ""]

        notation = spec.get("notation") or []
        if notation:
            lines += ["## Notation", "", "| Symbol | Meaning |", "|---|---|"]
            for entry in notation:
                if isinstance(entry, dict):
                    sym = str(entry.get("symbol", "")).replace("|", "\\|")
                    mean = str(entry.get("meaning", "")).replace("|", "\\|")
                    lines.append(f"| `{sym}` | {mean} |")
            lines.append("")

        equations = spec.get("equations") or []
        if equations:
            lines += ["## Mathematical Representation", ""]
            for eq in equations:
                if not isinstance(eq, dict):
                    continue
                eid = eq.get("id", "")
                latex = eq.get("latex", "")
                desc = eq.get("description", "")
                if eid:
                    lines.append(f"**({eid})**")
                if latex:
                    lines += ["", "$$", str(latex), "$$", ""]
                if desc:
                    lines += [str(desc), ""]

        pseudo = spec.get("pseudocode") or []
        if pseudo:
            lines += ["## Pseudocode", "", "```"]
            if isinstance(pseudo, list):
                for ln in pseudo:
                    lines.append(str(ln))
            else:
                lines.append(str(pseudo))
            lines += ["```", ""]

        complexity = spec.get("complexity")
        if isinstance(complexity, dict):
            lines += ["## Complexity", ""]
            t = complexity.get("time")
            s = complexity.get("space")
            if t:
                lines.append(f"- **Time:** {t}")
            if s:
                lines.append(f"- **Space:** {s}")
            lines.append("")

        return "\n".join(lines)
