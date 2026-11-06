"""
NoSkelDesignerAgent — skeleton-free (NoSkel-Agents) algorithm designer with
dynamic sub-agent spawning, following the DR_Template_v2 design constraints
(docs/DR_Template_v2.md): explicit code classification, verified-code reuse,
score plan and hyperparameter-selection policy.

Runs a multi-turn loop (up to MAX_ROUNDS) where each round the LLM either:
  - requests specialized analysis agents ("spawn_agents"), or
  - produces a final algorithm specification ("finalize_design").

Sub-agents are instantiated on-the-fly as DynamicAgent instances.
This agent produces only an algorithm SPECIFICATION (no code).
"""

import json
import logging

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)

_ROUND_TEMPLATE = """You are designing a novel dimensionality reduction algorithm from a blank canvas under DR_Template_v2.
You have complete freedom in mathematics and structure, but you must explicitly separate VERIFIED deterministic components from DYNAMIC code produced for this run.

DR_Template_v2 requirements:
- Prefer verified deterministic code for stable subroutines such as reusable neighborhood construction, eigensolvers, score calculation utilities, and benchmark algorithm components.
- Use dynamic code only for the genuinely new algorithmic composition introduced by this run.
- Make the intended local/global score usage explicit.
- Define a fixed hyperparameter-selection mechanism.

═══════════════════════ DATA CHARACTERIZATION ═══════════════════════
{data_char_json}

═══════════════════════ SUB-AGENT ANALYSIS GATHERED SO FAR ═══════════════════════
{sub_results_json}

═══════════════════════ DESIGN ROUND {round_num} / {max_rounds} ═══════════════════════
{round_instruction}

Respond with exactly one of these two JSON shapes:

OPTION A — Request specialized analysis sub-agents (not available in final round):
{{
  "action": "spawn_agents",
  "reasoning": "brief explanation of why you need this analysis before finalizing",
  "agents": [
    {{
      "role": "DescriptiveName",
      "system_message": "Complete, self-contained system message for this agent.",
      "task_prompt": "Complete task prompt with all necessary context embedded — the agent has no other information."
    }}
  ]
}}

OPTION B — Finalize your algorithm design (REQUIRED in final round):
{{
  "action": "finalize_design",
  "algorithm_name": "Descriptive name for the algorithm",
  "description": "2-3 sentence summary of the approach",
  "mathematical_formulation": "Step-by-step mathematical description",
  "key_innovations": ["innovation 1", "innovation 2"],
  "suggested_params": {{"param_name": default_value}},
  "code_classification": {{
    "verified_deterministic_components": ["string"],
    "dynamic_components": ["string"],
    "reusable_functional_algorithms": ["string"]
  }},
  "score_plan": {{
    "local_scores": ["trustworthiness", "continuity"],
    "global_scores": ["spearman", "stress"],
    "alignment_notes": "string"
  }},
  "hyperparameter_selection_mechanism": "string",
  "design_rationale": "Why this approach suits the observed data characteristics"
}}"""

_FORCE_FINALIZE_TEMPLATE = """Based on the data characterization and analysis gathered, produce a final algorithm specification now under DR_Template_v2.

DATA CHARACTERIZATION:
{data_char_json}

ANALYSIS GATHERED:
{sub_results_json}

Return JSON with these exact keys:
{{
  "action": "finalize_design",
  "algorithm_name": "...",
  "description": "...",
  "mathematical_formulation": "...",
  "key_innovations": [...],
  "suggested_params": {{...}},
  "code_classification": {{
    "verified_deterministic_components": [...],
    "dynamic_components": [...],
    "reusable_functional_algorithms": [...]
  }},
  "score_plan": {{
    "local_scores": [...],
    "global_scores": [...],
    "alignment_notes": "..."
  }},
  "hyperparameter_selection_mechanism": "...",
  "design_rationale": "..."
}}"""


class NoSkelDesignerAgent(BaseAgent):
    """Open-ended algorithm designer that spawns dynamic sub-agents as needed."""

    AGENT_NAME = "NoSkelDesigner"
    AGENT_ID = 11
    MAX_ROUNDS = 3

    def system_message(self) -> str:
        return (
            "You are designing a novel dimensionality reduction algorithm under DR_Template_v2. "
            "You have no constraints on mathematical approach, but you must explicitly classify "
            "verified deterministic components versus dynamic code, define score usage, and provide "
            "a fixed hyperparameter-selection policy. Your final output is an algorithm specification only. "
            "Always respond with valid JSON only. No markdown fences."
        )

    def build_prompt(self, context: dict) -> str:
        """Standard BaseAgent interface — satisfies the abstract method requirement.

        WARNING: This agent must be invoked via `run_design_loop()`, not `run()`.
        Calling `run()` executes only a single non-final round with no sub-agent spawning.
        """
        return self._build_round_prompt(context, [], 1, self.MAX_ROUNDS, is_last=False)

    def run_design_loop(self, data_characterization: dict, max_rounds: int = MAX_ROUNDS) -> dict:
        """
        Run the multi-turn design loop.

        Returns a finalize_design dict with keys:
          action, algorithm_name, description, mathematical_formulation,
          key_innovations, suggested_params, design_rationale.
        """
        sub_agent_results: list = []
        session_transcript: list = []

        for round_num in range(1, max_rounds + 1):
            is_last = (round_num == max_rounds)
            prompt = self._build_round_prompt(
                data_characterization, sub_agent_results, round_num, max_rounds, is_last
            )

            raw = self.llm.chat(
                system_message=self.system_message(),
                user_message=prompt,
            )
            parsed = self._parse_json(raw)
            if not isinstance(parsed, dict):
                # The LLM occasionally returns a bare JSON array (or null); treat
                # it as an empty round rather than crashing on .get().
                logger.warning(
                    "[NoSkelDesigner] Round %d returned non-dict JSON (%s); skipping.",
                    round_num, type(parsed).__name__,
                )
                parsed = {}
            action = parsed.get("action", "")

            session_transcript.append({
                "round": round_num,
                "action": action,
                "response": parsed,
            })

            if action == "finalize_design":
                parsed["_session_transcript"] = session_transcript
                return parsed

            if action == "spawn_agents" and not is_last:
                for spec in parsed.get("agents", []):
                    result = self._run_sub_agent(spec)
                    sub_agent_results.append({
                        "role": spec.get("role", "Unknown"),
                        "result": result,
                    })
                    logger.info("[NoSkelDesigner] Sub-agent '%s' completed", spec.get("role"))
            elif is_last:
                logger.warning("[NoSkelDesigner] Round %d reached without finalize_design — forcing", round_num)
                forced = self._force_finalize(data_characterization, sub_agent_results)
                forced["_session_transcript"] = session_transcript
                forced["_forced_finalize"] = True
                return forced

        # Safety net
        return {
            "action": "finalize_design",
            "algorithm_name": "Adaptive DR",
            "description": "Data-driven dimensionality reduction.",
            "mathematical_formulation": "Adaptive embedding via learned affinities.",
            "key_innovations": [],
            "suggested_params": {"n_components": 2},
            "design_rationale": "Safety fallback",
            "_session_transcript": session_transcript,
        }

    def _build_round_prompt(
        self,
        data_char: dict,
        sub_results: list,
        round_num: int,
        max_rounds: int,
        is_last: bool,
    ) -> str:
        data_json = json.dumps(data_char, indent=2, default=str)[:3000]
        sub_json = (
            json.dumps(sub_results, indent=2, default=str)[:4000]
            if sub_results
            else "(none yet — this is the first round)"
        )

        if is_last:
            instruction = (
                "This is the FINAL round. You MUST respond with 'finalize_design'. "
                "Produce your best DR_Template_v2-aligned specification now."
            )
        else:
            remaining = max_rounds - round_num
            instruction = (
                f"You have {remaining} more round(s) after this. "
                "You may request additional analysis agents OR finalize your design now if you have enough information."
            )

        return _ROUND_TEMPLATE.format(
            data_char_json=data_json,
            sub_results_json=sub_json,
            round_num=round_num,
            max_rounds=max_rounds,
            round_instruction=instruction,
        )

    def _run_sub_agent(self, spec: dict) -> dict:
        """Instantiate and run a DynamicAgent from a designer-specified spec."""
        from .dynamic_agent import DynamicAgent
        agent = DynamicAgent(
            self.llm,
            role=spec.get("role", "AnalysisAgent"),
            system_msg=spec.get("system_message", self.system_message()),
            task_prompt=spec.get("task_prompt", "Analyze and return JSON."),
        )
        try:
            return agent.run({})
        except Exception as exc:
            logger.warning("[NoSkelDesigner] Sub-agent '%s' failed: %s", spec.get("role"), exc)
            return {"error": str(exc)}

    def _force_finalize(self, data_char: dict, sub_results: list) -> dict:
        prompt = _FORCE_FINALIZE_TEMPLATE.format(
            data_char_json=json.dumps(data_char, indent=2, default=str)[:2000],
            sub_results_json=json.dumps(sub_results, indent=2, default=str)[:2000],
        )
        raw = self.llm.chat(system_message=self.system_message(), user_message=prompt)
        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        if "algorithm_name" not in parsed:
            parsed = {
                "action": "finalize_design",
                "algorithm_name": "Data-Driven DR v2",
                "description": "Algorithm derived from data characteristics and DR_Template_v2 constraints.",
                "mathematical_formulation": "Adaptive embedding.",
                "key_innovations": [],
                "suggested_params": {"n_components": 2},
                "code_classification": {
                    "verified_deterministic_components": [],
                    "dynamic_components": [],
                    "reusable_functional_algorithms": [],
                },
                "score_plan": {
                    "local_scores": ["trustworthiness", "continuity"],
                    "global_scores": ["spearman", "stress"],
                    "alignment_notes": "Auto-generated after exhausting design rounds.",
                },
                "hyperparameter_selection_mechanism": "Use provided defaults as the initial fixed policy.",
                "design_rationale": "Auto-generated after exhausting design rounds.",
            }
        return parsed
