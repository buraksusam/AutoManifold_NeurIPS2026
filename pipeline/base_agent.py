"""
Base class for all worker agents. Handles LLM communication, prompt formatting,
JSON response parsing, and skill usage tracking (PUA/Superpowers).
"""

import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional

from .llm_client import LLMClient

# Optional skill tracking - import if available
try:
    from .skill_tracker import track_skill, get_tracker
    SKILL_TRACKING_AVAILABLE = True
except ImportError:
    SKILL_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for every worker agent in the pipeline.

    Each agent is responsible for:
      1. Building a prompt from its template + runtime context.
      2. Sending the prompt to the LLM.
      3. Parsing the structured JSON response.
      4. Validating the response schema.
    """

    # Subclasses MUST set these
    AGENT_NAME: str = "BaseAgent"
    AGENT_ID: int = 0

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(f"agent.{self.AGENT_NAME}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        """
        Execute this agent's task.

        Args:
            context: Dict of all information the agent needs (data specs,
                     outputs from prior agents, current metrics, etc.)
        Returns:
            Parsed JSON dict — the agent's structured output.
        """
        # Track skill usage if available
        skill_ctx = None
        if SKILL_TRACKING_AVAILABLE:
            # Auto-detect skill based on agent type and context
            skill_info = self._detect_skill(context)
            if skill_info:
                skill_ctx = track_skill(
                    skill_name=skill_info["name"],
                    skill_type=skill_info["type"],
                    agent_name=self.AGENT_NAME,
                    agent_id=self.AGENT_ID,
                    parameters=skill_info.get("parameters"),
                )
                if skill_ctx is not None:
                    skill_ctx.__enter__()

        try:
            self.logger.info(f"[{self.AGENT_NAME}] Building prompt …")
            prompt = self.build_prompt(context)

            self.logger.info(f"[{self.AGENT_NAME}] Calling LLM …")
            raw_response = self.llm.chat(
                system_message=self.system_message(),
                user_message=prompt,
                images=context.get("_images"),  # for vision-capable models
            )

            self.logger.info(f"[{self.AGENT_NAME}] Parsing response …")
            parsed = self._parse_json(raw_response)

            # Coerce lists / other non-dict JSON to a dict when possible
            if parsed is not None and not isinstance(parsed, dict):
                self.logger.warning(
                    f"[{self.AGENT_NAME}] Expected JSON object but got "
                    f"{type(parsed).__name__}; attempting coercion …"
                )
                coerced = self._coerce_to_dict(parsed)
                if coerced is not None:
                    self.logger.info(
                        f"[{self.AGENT_NAME}] Coerced {type(parsed).__name__} → dict OK"
                    )
                    parsed = coerced

            if parsed is None or not isinstance(parsed, dict):
                self.logger.warning(
                    f"[{self.AGENT_NAME}] JSON parse failed, retrying with nudge …"
                )
                parsed = self._retry_parse(raw_response)

            if parsed is not None and isinstance(parsed, dict):
                # Guard: treat single-key {"error": ...} dicts as failures
                if list(parsed.keys()) == ["error"]:
                    self.logger.warning(
                        f"[{self.AGENT_NAME}] LLM returned error dict: {parsed.get('error', '')[:120]}"
                    )
                    parsed = {}
                else:
                    self._validate(parsed)
            else:
                self.logger.error(
                    f"[{self.AGENT_NAME}] Could not obtain valid JSON object after retry — returning {{}}."
                )
                parsed = {}

            return parsed
        finally:
            # End skill tracking
            if skill_ctx is not None:
                skill_ctx.__exit__(*sys.exc_info())

    def _detect_skill(self, context: dict) -> Optional[dict]:
        """
        Auto-detect which PUA/Superpowers skill is being used based on
        agent type and context. Override in subclasses for custom skill detection.
        """
        # Default detection based on agent type
        skill_map = {
            "DataCharacterization": {"name": "brainstorming", "type": "superpowers"},
            "EmbeddingDifficulty": {"name": "brainstorming", "type": "superpowers"},
            "AffinityDesigner": {"name": "brainstorming", "type": "superpowers"},
            "CostComposer": {"name": "brainstorming", "type": "superpowers"},
            "HyperparameterOpt": {"name": "systematic_debugging", "type": "superpowers"},
            "VisualizationEvaluator": {"name": "verification_before_completion", "type": "superpowers"},
            "NoveltyValidator": {"name": "systematic_debugging", "type": "superpowers"},
            "ErrorRecovery": {"name": "systematic_debugging", "type": "superpowers"},
        }

        # Check for PUA mode in context
        if context.get("pua_mode"):
            return {"name": "pua", "type": "pua", "parameters": {"mode": context.get("pua_mode")}}

        # Look up in skill map
        return skill_map.get(self.AGENT_NAME)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        """Build the user-message prompt from the template + context."""
        ...

    def system_message(self) -> str:
        """Default system message. Override in subclass if needed."""
        return (
            "You are a specialized AI agent in a dimensionality reduction "
            "algorithm design pipeline. Always respond with valid JSON only. "
            "Do not include markdown fences or explanatory text outside the JSON."
        )

    def _validate(self, output: dict) -> None:
        """Optional schema validation hook. Override per agent."""
        pass

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Optional[Any]:
        """Extract and parse JSON from an LLM response string.

        Returns a dict, list, or None.  Callers are responsible for
        coercing non-dict values via `_coerce_to_dict`.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` fences  (object OR array)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match_arr = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match_arr:
            try:
                return json.loads(match_arr.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding the outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Try finding the outermost [ ... ]
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(text[start_arr : end_arr + 1])
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _coerce_to_dict(parsed: Any) -> Optional[dict]:
        """Best-effort coercion of a non-dict JSON value to a dict.

        * Single-element list containing a dict → unwrap.
        * Multi-element list of dicts → merge (later keys overwrite).
        * Other list → wrap under ``{"items": <list>}``.
        """
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
            if all(isinstance(el, dict) for el in parsed) and parsed:
                merged: dict = {}
                for d in parsed:
                    merged.update(d)
                return merged
            return {"items": parsed}
        return None

    def _retry_parse(self, raw: str) -> Optional[dict]:
        """Ask the LLM to fix its own output into a valid JSON *object*."""
        # If there's nothing to fix, skip the retry
        if not raw or not raw.strip():
            return None
        # Send the full response — a truncated excerpt can't be repaired
        raw_excerpt = raw.strip()
        fix_prompt = (
            "The following text could not be parsed as a valid JSON **object** "
            "(i.e. it must start with `{` and end with `}`).\n"
            "If the text is a JSON array, merge all elements into a single "
            "JSON object.\n"
            "Return ONLY the corrected complete JSON object with no markdown, "
            "no code fences, no commentary:\n\n"
            f"{raw_excerpt}"
        )
        try:
            fixed = self.llm.chat(
                system_message="You are a JSON repair assistant. Output only a valid JSON object (not an array).",
                user_message=fix_prompt,
            )
        except Exception as exc:
            self.logger.error(f"[{self.AGENT_NAME}] Retry LLM call failed: {exc}")
            return None
        result = self._parse_json(fixed)
        # Apply coercion in case the retry still returns a list
        if result is not None and not isinstance(result, dict):
            result = self._coerce_to_dict(result)
        return result
