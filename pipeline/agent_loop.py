"""AgentLoop — autonomous LLM agent loop for Skills Mode v2.

Drives a single LLM conversation through the planner prompt + a registry of
tools (skills + compute tools + finalize). Termination is triggered by the
model calling ``finalize_design`` or by the iteration cap.

Design notes:

* This module owns the conversation messages (Anthropic-native content-block
  shape; ``LLMClient`` translates per-provider).
* Streaming + thinking + per-turn artifact persistence are handled here.
  ``SkillSession`` is preserved as the legacy phase-orchestrator state holder
  for non-agent paths and for backwards compatibility with its tests; this
  module does NOT depend on it.
* Per-iteration artifacts are written to ``<output_dir>/agent_turns/NN_iteration.json``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .llm_client import LLMClient
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Mutable state shared by every tool handler in a single agent run."""

    data: np.ndarray
    labels: Optional[np.ndarray]
    label_names: Optional[Dict[int, str]]
    primary_goal: str
    output_dir: str
    dr_engine: Any
    metrics_engine: Any
    clustering_engine: Any
    metrics_k: int = 10
    lof_neighbors: int = 20
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    code_attempts: List[Dict[str, Any]] = field(default_factory=list)
    code_attempt_count: int = 0
    baseline_profile: Optional[Dict[str, Any]] = None
    point_difficulty: Optional[List[str]] = None
    point_reliability: Optional[List[float]] = None
    final_report: Optional[Dict[str, Any]] = None
    done: bool = False


class AgentLoop:
    """Single-conversation autonomous agent loop driven by LLM tool-use."""

    MAX_ITERATIONS = 60
    MAX_NUDGES = 3

    def __init__(
        self,
        llm_client: LLMClient,
        planner_prompt: str,
        tool_registry: ToolRegistry,
        session_state: SessionState,
        log_callback: Optional[Callable[[str, str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        tool_call_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        skill_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        self.llm = llm_client
        self.planner_prompt = planner_prompt
        self.tools = tool_registry
        self.state = session_state
        self.log_callback = log_callback
        self.stream_callback = stream_callback
        self.thinking_callback = thinking_callback
        self.tool_call_callback = tool_call_callback
        self._skill_names = set(skill_names or [])
        self.verbose = verbose

        self._messages: List[Dict[str, Any]] = []
        self._nudge_count: int = 0
        self._last_tool_was_skill: bool = False
        self._last_skill_name: str = ""
        self._skill_output_grace_used: bool = False
        self.iteration_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_initial_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def run(self) -> Dict[str, Any]:
        for iteration in range(self.MAX_ITERATIONS):
            self.iteration_count = iteration + 1
            # Heartbeat: if the user only ever sees the previous iteration's
            # "done" line and never this "calling LLM" line for the next
            # iteration, the SDK call itself is the hang point — and the
            # client-level timeout in LLMClient will surface it as an error.
            self._emit_log(
                f"[agent] iteration {self.iteration_count} — calling LLM with "
                f"{len(self._messages)} messages",
            )

            iter_started = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            messages_before = len(self._messages)
            try:
                response = self.llm.chat_history(
                    system=self.planner_prompt,
                    messages=self._messages,
                    tools=self.tools.schemas(),
                    stream_callback=self._build_stream_relay(),
                    thinking_callback=self._build_thinking_relay(),
                    tool_use_callback=self._build_tool_use_relay(),
                )
            except Exception as exc:
                latency = time.perf_counter() - t0
                self._emit_log(f"[agent] LLM call failed: {exc}", level="error")
                self._write_turn_artifact({
                    "iteration_index": self.iteration_count,
                    "started_at": iter_started,
                    "latency_seconds": round(latency, 4),
                    "messages_before": messages_before,
                    "messages_after": messages_before,
                    "response_text": None,
                    "tool_calls": [],
                    "tool_results": [],
                    "error": str(exc),
                    "done": False,
                })
                raise
            latency = time.perf_counter() - t0

            if not isinstance(response, dict):
                # Backwards-compat: chat_history returned a string. Treat as text-only.
                response = {
                    "type": "text",
                    "text": str(response),
                    "tool_calls": [],
                    "stop_reason": "end_turn",
                    "raw_response": str(response),
                    "thinking_response": "",
                }

            tool_calls = response.get("tool_calls") or []
            response_text = response.get("text") or ""

            # Append assistant turn (Anthropic-native content blocks)
            self._append_assistant_turn(response)

            if not tool_calls:
                # Text-only response. Either we're already done, or we nudge.
                self._write_turn_artifact({
                    "iteration_index": self.iteration_count,
                    "started_at": iter_started,
                    "latency_seconds": round(latency, 4),
                    "messages_before": messages_before,
                    "messages_after": len(self._messages),
                    "response_text": response_text,
                    "thinking_response": response.get("thinking_response") or "",
                    "tool_calls": [],
                    "tool_results": [],
                    "stop_reason": response.get("stop_reason"),
                    "done": self.state.done,
                })
                if self.state.done:
                    break
                # Skill-output grace: if the previous tool was a skill, this
                # text-only turn is the skill's required JSON output and does
                # NOT count against the nudge budget. Re-prompt for the next
                # action once without incrementing the counter.
                if self._last_tool_was_skill and not self._skill_output_grace_used:
                    self._skill_output_grace_used = True
                    self._last_tool_was_skill = False
                    grace_msg = (
                        "Skill output received. Your next assistant turn MUST "
                        "include a tool_use block. "
                    )
                    if self._last_skill_name in {"synthesize-algorithm", "recover-failing-code"}:
                        grace_msg += (
                            "Because you just produced executable code, the "
                            "required next tool is `validate_custom_code` — "
                            "pass the synthesized `custom_embedding` source as "
                            "its `code` argument."
                        )
                    elif self._last_skill_name == "interpret-forensics":
                        grace_msg += (
                            "Because forensic interpretation is the final "
                            "workflow step, the required next tool is "
                            "`finalize_design`."
                        )
                    else:
                        grace_msg += (
                            "Invoke the next skill, a compute tool, or "
                            "`finalize_design` if the design is complete."
                        )
                    self._messages.append({"role": "user", "content": grace_msg})
                    continue
                if self._nudge_count >= self.MAX_NUDGES:
                    self._emit_log(
                        "[agent] no tool call after %d nudges — terminating"
                        % self._nudge_count,
                        level="warning",
                    )
                    break
                self._nudge_count += 1
                self._messages.append({
                    "role": "user",
                    "content": (
                        "You produced no tool call. If the design is complete, "
                        "call `finalize_design` with a final_report summary. "
                        "Otherwise continue with the next skill or compute tool."
                    ),
                })
                continue

            # A real tool call resets both the nudge counter and the
            # skill-output grace bookkeeping.
            self._nudge_count = 0
            self._skill_output_grace_used = False
            self._last_tool_was_skill = any(
                (tc.get("name") or "") in self._skill_names for tc in tool_calls
            )
            self._last_skill_name = next(
                (
                    (tc.get("name") or "")
                    for tc in tool_calls
                    if (tc.get("name") or "") in self._skill_names
                ),
                "",
            )

            # Dispatch each tool call sequentially
            tool_results: List[Dict[str, Any]] = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments") or {}
                tc_id = tc.get("id", "")
                self._emit_log(
                    f"[agent] tool: {name}({_short_args(args)})",
                )
                if self.tool_call_callback is not None:
                    try:
                        self.tool_call_callback("call", {
                            "tool_name": name,
                            "arguments": args,
                            "iteration": self.iteration_count,
                        })
                    except Exception as cb_exc:
                        logger.debug("tool_call_callback raised: %s", cb_exc)

                result = self.tools.dispatch(name, args, self.state)
                tool_results.append({
                    "tool_use_id": tc_id,
                    "name": name,
                    "result": result,
                })
                if self.tool_call_callback is not None:
                    try:
                        summary = self._summarize_tool_result(name, result)
                        self.tool_call_callback("result", {
                            "tool_name": name,
                            "result_summary": summary,
                            "iteration": self.iteration_count,
                        })
                    except Exception as cb_exc:
                        logger.debug("tool_call_callback raised: %s", cb_exc)
                if name == "finalize_design":
                    # state.done is set by the handler; we still loop through
                    # remaining tool_calls (if any) before stopping.
                    pass

            # Append tool results as a single user message
            self._append_tool_results(tool_results)
            self._write_turn_artifact({
                "iteration_index": self.iteration_count,
                "started_at": iter_started,
                "latency_seconds": round(latency, 4),
                "messages_before": messages_before,
                "messages_after": len(self._messages),
                "response_text": response_text,
                "thinking_response": response.get("thinking_response") or "",
                "tool_calls": [
                    {"id": tc.get("id"), "name": tc.get("name"),
                     "arguments": tc.get("arguments")}
                    for tc in tool_calls
                ],
                "tool_results": tool_results,
                "stop_reason": response.get("stop_reason"),
                "done": self.state.done,
            })
            self._emit_log(
                f"[agent] iteration {self.iteration_count} done; "
                f"{len(tool_results)} tool result(s)",
            )

            if self.state.done:
                break
        else:
            self._emit_log(
                "[agent] hit max iterations without finalize_design",
                level="warning",
            )
            if self.state.final_report is None:
                self.state.final_report = {
                    "warning": "max iterations reached",
                    "iteration_count": self.iteration_count,
                }

        return self.state.final_report or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_assistant_turn(self, response: Dict[str, Any]) -> None:
        blocks: List[Dict[str, Any]] = []
        text = response.get("text") or ""
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in response.get("tool_calls") or []:
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "input": tc.get("arguments") or {},
            })
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        msg: Dict[str, Any] = {"role": "assistant", "content": blocks}
        # DeepSeek thinking mode requires the assistant's `reasoning_content`
        # to be round-tripped on subsequent turns when a tool call was
        # involved. We store it as a sibling field; the LLMClient's
        # provider-specific outgoing translation is responsible for emitting
        # it on the wire (DeepSeek path only — Anthropic handles thinking
        # signatures internally via the SDK).
        rc = response.get("reasoning_content")
        if rc:
            msg["reasoning_content"] = rc
        self._messages.append(msg)

    def _append_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        if not tool_results:
            return
        blocks: List[Dict[str, Any]] = []
        for tr in tool_results:
            result = tr["result"]
            if isinstance(result, (dict, list)):
                content = json.dumps(result, default=str)
            else:
                content = str(result)
            blocks.append({
                "type": "tool_result",
                "tool_use_id": tr["tool_use_id"],
                "content": content,
            })
        self._messages.append({"role": "user", "content": blocks})

    # --- streaming relays ---

    def _build_stream_relay(self) -> Optional[Callable[[str], None]]:
        if self.stream_callback is None and not self.verbose:
            return None
        verbose = self.verbose
        external = self.stream_callback

        def _relay(chunk: str) -> None:
            if not chunk:
                return
            if verbose:
                print(chunk, end="", flush=True)
            if external is not None:
                try:
                    external(chunk)
                except Exception as exc:
                    logger.debug("stream_callback raised: %s", exc)
        return _relay

    def _build_thinking_relay(self) -> Optional[Callable[[str], None]]:
        if self.thinking_callback is None:
            return None
        external = self.thinking_callback
        verbose = self.verbose

        def _relay(chunk: str) -> None:
            if not chunk:
                return
            if verbose:
                print(chunk, end="", flush=True)
            try:
                external(chunk)
            except Exception as exc:
                logger.debug("thinking_callback raised: %s", exc)
        return _relay

    def _build_tool_use_relay(self) -> Optional[Callable[[str, str], None]]:
        if self.tool_call_callback is None:
            return None

        def _relay(name: str, partial: str) -> None:
            # We don't fire tool_call_callback per-delta — that fires when the
            # full tool call is dispatched. This relay just keeps the seam open.
            return None
        return _relay

    # --- artifact writing ---

    def _write_turn_artifact(self, payload: Dict[str, Any]) -> None:
        try:
            turns_dir = Path(self.state.output_dir) / "agent_turns"
            os.makedirs(turns_dir, exist_ok=True)
            filename = f"{payload['iteration_index']:02d}_iteration.json"
            final_path = turns_dir / filename
            tmp_path = turns_dir / (filename + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, final_path)
        except Exception:
            logger.exception("AgentLoop failed to write per-turn artifact")

    # --- result summarization for UI ---

    def _summarize_tool_result(self, name: str, result: Any) -> Any:
        """Produce a UI-safe summary of a tool result.

        The full result still goes into the per-iteration artifact JSON; this
        method is used solely for the live ``tool_call_callback`` so the SSE
        payload does not blow up with embedding arrays / metric matrices.
        """
        # Non-dict results: return a short string repr.
        if not isinstance(result, dict):
            try:
                s = json.dumps(result, default=str)
            except Exception:
                s = str(result)
            if len(s) > 1000:
                s = s[:1000] + "...(truncated)"
            return s

        # Skill-tool results (handler returns skill_name + skill_body, etc.).
        if name in self._skill_names or ("skill_name" in result and "skill_body" in result):
            body = result.get("skill_body") or ""
            return {
                "skill_invoked": True,
                "task_context": (result.get("task_context_echo") or "")[:300],
                "body_chars": len(body),
            }

        if name == "validate_custom_code":
            emb = result.get("embedding")
            try:
                emb_shape = list(np.asarray(emb).shape) if emb is not None else None
            except Exception:
                emb_shape = None
            return {
                "ok": bool(result.get("ok")),
                "error_message": result.get("error_message"),
                "embedding_shape": emb_shape,
                "embedding_handle": result.get("embedding_handle"),
            }

        if name == "compute_baseline_calibration":
            return {
                "winner": result.get("winner"),
                "winner_score": result.get("winner_score"),
                "intent": result.get("intent"),
            }

        if name == "finalize_design":
            report = result.get("final_report") or {}
            keys = list(report.keys()) if isinstance(report, dict) else []
            return {
                "finalized": True,
                "algorithm_name": result.get("acknowledged"),
                "final_report_keys": keys,
            }

        # Generic compute tools: ship as-is if small, else top-level keys.
        try:
            blob = json.dumps(result, default=str)
        except Exception:
            blob = str(result)
        if len(blob) <= 2048:
            return result
        return {
            "summary_keys": list(result.keys()),
            "size_chars": len(blob),
        }

    # --- log helper ---

    def _emit_log(self, line: str, level: str = "info") -> None:
        if self.verbose:
            print(line, flush=True)
        if self.log_callback is not None:
            try:
                self.log_callback(line, level)
            except Exception:
                logger.debug("log_callback raised; ignoring.", exc_info=True)


def _short_args(args: Dict[str, Any], max_len: int = 200) -> str:
    """Render args dict as a short single-line string for log breadcrumbs."""
    try:
        s = json.dumps(args, default=str)
    except Exception:
        s = str(args)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
