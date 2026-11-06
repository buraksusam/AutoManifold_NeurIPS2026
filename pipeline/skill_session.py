"""SkillSession — drives a single accumulating LLM conversation across pipeline phases.

A "skill" is a markdown file with YAML frontmatter (name, phase, id, inputs,
outputs_schema_summary) and a `str.format`-style template body. The session
loads all skills from a directory, formats their bodies with render vars,
appends turns to a single accumulating message history, calls
`llm.chat_history`, persists the full transcript, and returns parsed dicts.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_agent import BaseAgent
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SkillFile:
    name: str
    phase: int
    id: str
    inputs: List[str] = field(default_factory=list)
    outputs_schema_summary: str = ""
    body: str = ""
    path: str = ""


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_yaml_value(raw: str) -> Any:
    """Parse a single flat YAML-ish value (string, bool, list of strings)."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        items = [it.strip() for it in inner.split(",")]
        cleaned: List[str] = []
        for it in items:
            it = it.strip()
            if (it.startswith("'") and it.endswith("'")) or (
                it.startswith('"') and it.endswith('"')
            ):
                it = it[1:-1]
            cleaned.append(it)
        return cleaned
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        return raw[1:-1]
    return raw


def _parse_skill_file(path: Path) -> SkillFile:
    """Parse a markdown skill file into a SkillFile dataclass."""
    text = path.read_text()
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(
            f"Skill file '{path}' missing YAML frontmatter (must begin with '---')."
        )
    fm_text, body = m.group(1), m.group(2)

    parsed: Dict[str, Any] = {}
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        parsed[key.strip()] = _parse_yaml_value(val)

    if "name" not in parsed or not parsed["name"]:
        raise ValueError(
            f"Skill file '{path}' missing required frontmatter field: 'name'"
        )

    name = str(parsed["name"])
    phase_raw = parsed.get("phase", 0)
    try:
        phase = int(phase_raw) if not isinstance(phase_raw, bool) else 0
    except (TypeError, ValueError):
        phase = 0
    id_ = str(parsed.get("id", ""))
    inputs = parsed.get("inputs", []) or []
    if not isinstance(inputs, list):
        inputs = []
    outputs_schema_summary = str(parsed.get("outputs_schema_summary", ""))

    return SkillFile(
        name=name,
        phase=phase,
        id=id_,
        inputs=list(inputs),
        outputs_schema_summary=outputs_schema_summary,
        body=body,
        path=str(path),
    )


class _RetryShim(BaseAgent):
    """Minimal BaseAgent subclass to access _retry_parse without inheriting agent state."""

    AGENT_NAME = "SkillSessionRetry"
    AGENT_ID = -1

    def build_prompt(self, context: dict) -> str:
        return ""


class SkillSession:
    """Drives a single accumulating LLM conversation through skill invocations."""

    SOFT_TOKEN_CAP = 150_000

    def __init__(
        self,
        llm_client: LLMClient,
        skills_dir: str,
        system_prompt_path: str,
        output_dir: str,
        verbose: bool = True,
        log_callback: Optional[Callable[[str, str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.llm = llm_client
        self.skills_dir = Path(skills_dir)
        if not self.skills_dir.is_dir():
            raise ValueError(f"Skills directory not found: {self.skills_dir}")
        self.output_dir = Path(output_dir)
        self.transcript_path = self.output_dir / "skills_session_transcript.json"
        self._messages: List[Dict[str, Any]] = []
        self.verbose = verbose
        self.log_callback = log_callback
        self._stream_callback = stream_callback
        self._thinking_callback = thinking_callback

        # Per-session invocation counters (Feature C)
        self._invocation_count: int = 0
        self._per_skill_count: Dict[str, int] = {}

        # Load callable skills from the skills directory (skip system prompt id == "00").
        self.skills: Dict[str, SkillFile] = {}
        for p in sorted(self.skills_dir.glob("*.md")):
            sf = _parse_skill_file(p)
            if sf.id == "00":
                continue
            if sf.name in self.skills:
                raise ValueError(f"Duplicate skill name '{sf.name}' in {p}")
            self.skills[sf.name] = sf

        # Load the system prompt.
        sys_path = Path(system_prompt_path)
        if not sys_path.exists():
            raise ValueError(f"System prompt file not found: {sys_path}")
        sys_skill = _parse_skill_file(sys_path)
        self.system_prompt = sys_skill.body.strip()

    def token_estimate(self) -> int:
        return sum(len(m.get("content", "")) for m in self._messages) // 4

    # ------------------------------------------------------------------
    # Logging helpers (Feature A)
    # ------------------------------------------------------------------

    def _emit(self, line: str, level: str = "info") -> None:
        """Print a [skill] breadcrumb (when verbose) and dispatch to log_callback."""
        if self.verbose:
            print(line, flush=True)
        if self.log_callback is not None:
            try:
                self.log_callback(line, level)
            except Exception:
                logger.debug("SkillSession log_callback raised; ignoring.", exc_info=True)

    @staticmethod
    def _id_name(skill: SkillFile) -> str:
        """Return e.g. '04_affinity_design' (or just the name if id is empty)."""
        if skill.id:
            return f"{skill.id}_{skill.name}"
        return skill.name

    # ------------------------------------------------------------------
    # Per-turn artifact helpers (Feature C)
    # ------------------------------------------------------------------

    def _artifact_filename(self, skill: SkillFile, attempt: int) -> str:
        """Build the filename for a per-turn artifact.

        attempt is the per-skill 1-based invocation number. The first call is
        unsuffixed; subsequent calls get '_attempt_<N>'.
        """
        base = self._id_name(skill)
        if attempt <= 1:
            return f"{base}.json"
        return f"{base}_attempt_{attempt}.json"

    def _write_turn_artifact(self, payload: Dict[str, Any], filename: str) -> None:
        """Atomically write a per-turn artifact JSON to <output_dir>/skill_turns/<filename>."""
        turns_dir = self.output_dir / "skill_turns"
        os.makedirs(turns_dir, exist_ok=True)
        final_path = turns_dir / filename
        tmp_path = turns_dir / (filename + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, final_path)

    def invoke_skill(
        self, skill_name: str, render_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        if skill_name not in self.skills:
            raise KeyError(
                f"unknown skill '{skill_name}' (known: {list(self.skills)})"
            )
        skill = self.skills[skill_name]

        try:
            user_text = skill.body.format(**render_vars)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "?"
            raise KeyError(
                f"skill '{skill_name}' template requires render var '{missing}' which was not provided"
            ) from exc

        # Bookkeeping for breadcrumbs and artifact (Features A + C)
        self._invocation_count += 1
        invocation_index = self._invocation_count
        self._per_skill_count[skill_name] = self._per_skill_count.get(skill_name, 0) + 1
        skill_invocation_index = self._per_skill_count[skill_name]
        artifact_filename = self._artifact_filename(skill, skill_invocation_index)
        id_name = self._id_name(skill)

        messages_before = len(self._messages)
        kb_before = self.token_estimate() / 256.0  # tokens → KB (≈ chars/4 → /256)
        turns_before = messages_before // 2

        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._emit(
            f"[skill] → invoking {id_name}  "
            f"(history: {turns_before} turns, ~{kb_before:.1f} KB)",
            "info",
        )

        prospective = self._messages + [{"role": "user", "content": user_text}]

        # Soft-cap check operates on what we're about to send
        estimate = sum(len(m.get("content", "")) for m in prospective) // 4
        if estimate > self.SOFT_TOKEN_CAP:
            logger.warning(
                "SkillSession cumulative token estimate (%d) exceeds soft cap (%d); sending anyway.",
                estimate, self.SOFT_TOKEN_CAP,
            )

        # Build the streaming relay (only when an external callback is set,
        # so default behavior is bit-identical to the non-streaming path).
        # The begin marker is deferred until the first text chunk arrives so
        # that — when extended thinking is active — the thinking-begin marker
        # appears before the stream-begin marker in the log.
        streaming_active = self._stream_callback is not None
        stream_started = False  # whether stream-begin has been emitted yet

        if streaming_active:
            def _stream_relay(chunk: str) -> None:
                nonlocal stream_started
                if not chunk:
                    return
                if not stream_started:
                    self._emit(f"[skill stream begin] {id_name}", "info")
                    stream_started = True
                if self.verbose:
                    # Inline write — no extra newline, so chunks concatenate.
                    print(chunk, end="", flush=True)
                if self._stream_callback is not None:
                    try:
                        self._stream_callback(chunk)
                    except Exception:
                        logger.debug(
                            "SkillSession stream_callback raised; ignoring.",
                            exc_info=True,
                        )
        else:
            _stream_relay = None  # type: ignore[assignment]

        # Build the thinking-content relay. We always pass *some* callable
        # when a thinking_callback is set so the LLM client knows to consume
        # reasoning deltas. Chunks are accumulated for the per-turn artifact.
        thinking_active = self._thinking_callback is not None
        thinking_chunks: List[str] = []
        thinking_started = False  # whether we have actually emitted any thinking output

        if thinking_active:
            def _thinking_relay(chunk: str) -> None:
                nonlocal thinking_started
                if not chunk:
                    return
                if not thinking_started:
                    # Defer the begin marker until the first chunk actually arrives,
                    # so non-thinking models don't spam empty thinking blocks.
                    self._emit(f"[skill thinking begin] {id_name}", "info")
                    thinking_started = True
                if self.verbose:
                    print(chunk, end="", flush=True)
                thinking_chunks.append(chunk)
                if self._thinking_callback is not None:
                    try:
                        self._thinking_callback(chunk)
                    except Exception:
                        logger.debug(
                            "SkillSession thinking_callback raised; ignoring.",
                            exc_info=True,
                        )
        else:
            _thinking_relay = None  # type: ignore[assignment]

        t0 = time.perf_counter()
        try:
            raw = self.llm.chat_history(
                system=self.system_prompt,
                messages=prospective,
                stream_callback=_stream_relay,
                thinking_callback=_thinking_relay,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            err_msg = str(exc)
            short = err_msg if len(err_msg) <= 120 else err_msg[:117] + "..."
            logger.error("SkillSession.invoke_skill('%s') LLM call failed: %s", skill_name, exc)
            if thinking_started:
                if self.verbose:
                    print("", flush=True)
                self._emit(
                    f"[skill thinking end] {id_name}  (failed after {latency:.2f}s)",
                    "error",
                )
            if stream_started:
                self._emit(
                    f"[skill stream end] {id_name}  (failed after {latency:.2f}s)",
                    "error",
                )
            elif streaming_active:
                # Streaming was set up but no chunks arrived — still close the block.
                self._emit(
                    f"[skill stream end] {id_name}  (failed after {latency:.2f}s, no chunks)",
                    "error",
                )
            self._emit(
                f"[skill] ✗ {id_name}  LLM error after {latency:.1f}s: {short}",
                "error",
            )
            # Write artifact even on LLM exception (Feature C rule 2)
            try:
                payload = {
                    "skill_id": skill.id,
                    "skill_name": skill.name,
                    "skill_file": skill.path,
                    "invocation_index": invocation_index,
                    "skill_invocation_index": skill_invocation_index,
                    "started_at": started_at,
                    "latency_seconds": round(latency, 4),
                    "messages_before": messages_before,
                    "messages_after": messages_before,  # no commit on failure
                    "history_turns_before": turns_before,
                    "history_turns_after": turns_before,
                    "render_vars": render_vars,
                    "rendered_prompt": user_text,
                    "raw_response": None,
                    "parsed_response": None,
                    "parse_status": "llm_exception",
                    "streamed": streaming_active,
                    "thinking_response": "".join(thinking_chunks) if thinking_chunks else None,
                    "thinking_chars": sum(len(c) for c in thinking_chunks),
                    "reasoning_effort": getattr(self.llm.config, "reasoning_effort", None),
                    "error": err_msg,
                }
                self._write_turn_artifact(payload, artifact_filename)
            except Exception:
                logger.exception(
                    "SkillSession failed to write per-turn artifact for '%s'",
                    skill_name,
                )
            raise

        latency = time.perf_counter() - t0

        # Only commit the turn pair AFTER the call succeeds
        self._messages = prospective
        self._messages.append({"role": "assistant", "content": raw})
        self._persist_transcript()

        messages_after = len(self._messages)
        turns_after = messages_after // 2
        raw_str = raw if isinstance(raw, str) else str(raw)
        response_kb = len(raw_str) / 1024.0

        # Emit thinking-end first (it streams BEFORE the final answer).
        # Note: latency here covers the whole LLM call, not just the thinking
        # portion — the SDK doesn't expose a cheap thinking/answer boundary.
        if thinking_started:
            if self.verbose:
                print("", flush=True)
            think_chars = sum(len(c) for c in thinking_chunks)
            think_kb = think_chars / 1024.0
            self._emit(
                f"[skill thinking end] {id_name}  "
                f"({think_kb:.1f} KB in {latency:.2f}s)",
                "info",
            )

        if streaming_active:
            # Newline after the streamed body so the trailing chunk doesn't
            # run into the end marker on stdout.
            if self.verbose:
                print("", flush=True)
            self._emit(
                f"[skill stream end] {id_name}  "
                f"({response_kb:.1f} KB in {latency:.2f}s)",
                "info",
            )

        parsed = BaseAgent._parse_json(raw)
        if parsed is not None and not isinstance(parsed, dict):
            parsed = BaseAgent._coerce_to_dict(parsed)

        parse_status = "ok"
        if parsed is None or not isinstance(parsed, dict):
            logger.warning(
                "SkillSession could not parse response for skill '%s'; retrying via _retry_parse",
                skill_name,
            )
            self._emit(
                f"[skill] ⚠ {id_name}  unparseable response — retried via _retry_parse",
                "warning",
            )
            shim = _RetryShim(self.llm)
            parsed = shim._retry_parse(raw) or {}
            parse_status = "retried_ok" if isinstance(parsed, dict) and parsed else "retried_ok"

        if not isinstance(parsed, dict):
            parsed = {}

        result: Dict[str, Any]
        if list(parsed.keys()) == ["error"]:
            logger.warning(
                "SkillSession skill '%s' returned error dict: %s",
                skill_name,
                str(parsed.get("error", ""))[:120],
            )
            self._emit(
                f"[skill] ⚠ {id_name}  returned empty dict",
                "warning",
            )
            parse_status = "error_dict_collapsed"
            result = {}
        else:
            if parse_status == "ok":
                self._emit(
                    f"[skill] ← {id_name}  ok in {latency:.2f}s  "
                    f"(response: {response_kb:.1f} KB, {turns_after} turns total)",
                    "info",
                )
            else:
                # retried path — emit a confirming post line
                self._emit(
                    f"[skill] ← {id_name}  ok in {latency:.2f}s  "
                    f"(response: {response_kb:.1f} KB, {turns_after} turns total) "
                    f"[parse: {parse_status}]",
                    "info",
                )
            result = parsed

        # Write per-turn artifact (Feature C)
        try:
            payload = {
                "skill_id": skill.id,
                "skill_name": skill.name,
                "skill_file": skill.path,
                "invocation_index": invocation_index,
                "skill_invocation_index": skill_invocation_index,
                "started_at": started_at,
                "latency_seconds": round(latency, 4),
                "messages_before": messages_before,
                "messages_after": messages_after,
                "history_turns_before": turns_before,
                "history_turns_after": turns_after,
                "render_vars": render_vars,
                "rendered_prompt": user_text,
                "raw_response": raw_str,
                "parsed_response": parsed if parsed else (None if parse_status == "error_dict_collapsed" else parsed),
                "parse_status": parse_status,
                "streamed": streaming_active,
                "thinking_response": "".join(thinking_chunks) if thinking_chunks else None,
                "thinking_chars": sum(len(c) for c in thinking_chunks),
                "reasoning_effort": getattr(self.llm.config, "reasoning_effort", None),
                "error": None,
            }
            self._write_turn_artifact(payload, artifact_filename)
        except Exception:
            logger.exception(
                "SkillSession failed to write per-turn artifact for '%s'",
                skill_name,
            )

        return result

    def _persist_transcript(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "system_prompt": self.system_prompt,
            "messages": self._messages,
            "token_estimate": self.token_estimate(),
        }
        self.transcript_path.write_text(
            json.dumps(payload, indent=2, default=str)
        )
