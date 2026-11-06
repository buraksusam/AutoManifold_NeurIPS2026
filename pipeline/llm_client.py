"""
Unified LLM client supporting OpenAI, Anthropic, and Google backends.
Handles retries, rate-limits, and image (vision) inputs.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around LLM APIs with retry logic."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self):
        timeout = float(getattr(self.config, "timeout", 600) or 600)

        if self.config.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=self.config.api_key, timeout=timeout)

        elif self.config.provider == "deepseek":
            # DeepSeek is OpenAI-compatible — reuse the OpenAI SDK with
            # DeepSeek's base URL.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.deepseek.com",
                timeout=timeout,
            )

        elif self.config.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.config.api_key, timeout=timeout)

        elif self.config.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)
            # model is created without system_instruction here;
            # we rebuild it per-call so we can pass the agent's system message
            self._client = genai  # store the module, not the model
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        system_message: str,
        user_message: str,
        images: Optional[list[str | Path]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Send a single-turn prompt and return the raw text response.

        Backward-compatible wrapper around :meth:`chat_history`.

        Args:
            system_message: System/role instruction.
            user_message:   The user prompt.
            images:         Optional list of image file paths for vision models.
            max_retries:    Retry count on transient failures.

        Returns:
            Raw string response from the model.
        """
        user_msg: dict = {"role": "user", "content": user_message}
        if images:
            user_msg["images"] = images
        return self.chat_history(
            system=system_message,
            messages=[user_msg],
            max_retries=max_retries,
        )

    def chat_history(
        self,
        system: str,
        messages: list[dict],
        max_retries: int = 3,
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        tools: Optional[list[dict]] = None,
        tool_use_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Send a multi-turn conversation and return the raw text response.

        Args:
            system:            System/role instruction.
            messages:          List of {"role": "user"|"assistant", "content": ...,
                               "images": Optional[list]} dicts. When tools are in
                               play, content may be a list of Anthropic-style
                               content blocks (text/tool_use/tool_result).
            max_retries:       Retry count on transient failures.
            stream_callback:   Optional callback invoked with each FINAL-answer
                               text delta as it streams.
            thinking_callback: Optional callback invoked with each chain-of-thought
                               (reasoning) text delta when the model emits one.
            tools:             Optional list of tool schemas in our internal
                               (Anthropic-native) format:
                               ``[{"name": ..., "description": ...,
                                   "input_schema": {...}}]``.
                               When provided, the return type changes from str
                               to a structured dict (see below).
            tool_use_callback: Optional callback ``(tool_name, partial_args)``
                               invoked while the model streams partial tool-input
                               JSON. Useful for UI breadcrumbs.

        Returns:
            When ``tools is None``: raw string response (the FINAL answer only).
            When ``tools is not None``: a dict::

                {
                    "type": "tool_use" | "text" | "stop",
                    "text": <str: accumulated text>,
                    "tool_calls": [
                        {"id": str, "name": str, "arguments": dict},
                        ...
                    ],
                    "stop_reason": <str>,
                    "raw_response": <str: full assistant text>,
                    "thinking_response": <str: accumulated thinking>,
                }
        """
        for attempt in range(1, max_retries + 1):
            try:
                if self.config.provider in ("openai", "deepseek"):
                    return self._chat_openai_history(
                        system, messages,
                        stream_callback=stream_callback,
                        thinking_callback=thinking_callback,
                        tools=tools,
                        tool_use_callback=tool_use_callback,
                    )
                elif self.config.provider == "anthropic":
                    return self._chat_anthropic_history(
                        system, messages,
                        stream_callback=stream_callback,
                        thinking_callback=thinking_callback,
                        tools=tools,
                        tool_use_callback=tool_use_callback,
                    )
                elif self.config.provider == "google":
                    if tools is not None:
                        return self._chat_google_history_tools(
                            system, messages, tools,
                            stream_callback=stream_callback,
                            tool_use_callback=tool_use_callback,
                        )
                    # TODO: Gemini 2.5 supports `thinking_config` — not in scope here.
                    return self._chat_google_history(
                        system, messages, stream_callback=stream_callback
                    )
            except Exception as exc:
                logger.warning(
                    f"LLM call attempt {attempt}/{max_retries} failed: {exc}"
                )
                if attempt == max_retries:
                    raise
                time.sleep(2**attempt)

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    # Models that use max_completion_tokens instead of max_tokens,
    # and do not support response_format=json_object via the legacy param.
    _COMPLETION_TOKENS_MODELS = ("o1", "o3", "o4", "gpt-5", "o1-mini", "o1-preview")

    def _uses_completion_tokens(self) -> bool:
        m = self.config.model.lower()
        return any(m.startswith(prefix) for prefix in self._COMPLETION_TOKENS_MODELS)

    # Anthropic models that have deprecated the `temperature` parameter.
    # Claude 4.x (opus-4-7, sonnet-4-6, haiku-4-5, etc.) reject `temperature`.
    _ANTHROPIC_NO_TEMPERATURE_MODELS = (
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
    )

    def _anthropic_temperature_deprecated(self) -> bool:
        m = self.config.model.lower()
        return any(m.startswith(prefix) for prefix in self._ANTHROPIC_NO_TEMPERATURE_MODELS)

    # Claude 4.x models also reject assistant-message prefill — we coax JSON via
    # the system prompt + extract the object from the response.
    _ANTHROPIC_NO_PREFILL_MODELS = (
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
    )

    def _anthropic_no_prefill(self) -> bool:
        m = self.config.model.lower()
        return any(m.startswith(prefix) for prefix in self._ANTHROPIC_NO_PREFILL_MODELS)

    # Anthropic models that support extended thinking (chain-of-thought streaming).
    _ANTHROPIC_THINKING_MODELS = (
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",   # legacy thinking-capable model
    )

    def _anthropic_supports_thinking(self) -> bool:
        m = self.config.model.lower()
        return any(m.startswith(prefix) for prefix in self._ANTHROPIC_THINKING_MODELS)

    # Map of reasoning_effort → Anthropic thinking budget_tokens.
    _ANTHROPIC_EFFORT_BUDGET = {
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Return the first balanced top-level JSON object (or array) from text.

        Handles leading prose, markdown code fences, and trailing commentary.
        Falls back to the raw text if no object is found (callers then surface
        the JSON parse error).
        """
        if not text:
            return text
        start = None
        opener = None
        closer = None
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                opener = ch
                closer = "}" if ch == "{" else "]"
                break
        if start is None:
            return text
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]  # unbalanced — let caller's JSON parser report it

    # DEPRECATED: superseded by _chat_openai_history; will be removed in next release
    def _chat_openai(
        self, system_msg: str, user_msg: str, images: Optional[list] = None
    ) -> str:
        messages = [{"role": "system", "content": system_msg}]

        # Build user content (text + optional images)
        content_parts = [{"type": "text", "text": user_msg}]
        if images:
            for img_path in images:
                b64 = self._encode_image(img_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        messages.append({"role": "user", "content": content_parts})

        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        # Newer models (o1, o3, o4, gpt-5.x) use max_completion_tokens
        if self._uses_completion_tokens():
            if self.config.max_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.max_tokens
            # gpt-5.x is a standard chat model — force JSON output
            # True reasoning models (o1/o3/o4) do not support response_format
            m = self.config.model.lower()
            _no_json_format = ("o1", "o3", "o4")
            if not any(m.startswith(p) for p in _no_json_format):
                kwargs["response_format"] = {"type": "json_object"}
        else:
            if self.config.max_tokens is not None:
                kwargs["max_tokens"] = self.config.max_tokens
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    # DEPRECATED: superseded by _chat_anthropic_history; will be removed in next release
    def _chat_anthropic(
        self, system_msg: str, user_msg: str, images: Optional[list] = None
    ) -> str:
        content_parts = [{"type": "text", "text": user_msg}]
        if images:
            for img_path in images:
                b64 = self._encode_image(img_path)
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )

        # Anthropic requires streaming for requests that may exceed 10 minutes,
        # which is likely at our default max_tokens=64000. Use messages.stream
        # and accumulate the text deltas.
        # Claude 4.x models no longer support assistant-message prefill;
        # fall back to instruction-only JSON coaxing for those.
        use_prefill = not self._anthropic_no_prefill()
        messages: list = [{"role": "user", "content": content_parts}]
        if use_prefill:
            # Prefill forces Claude to start (and complete) a JSON object
            messages.append({"role": "assistant", "content": "{"})

        stream_kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 64000,  # default = Sonnet 4.x native cap
            "system": system_msg,
            "messages": messages,
        }
        # Newer Claude models (e.g. claude-opus-4-7, claude-sonnet-4-6) deprecate
        # the `temperature` parameter — only pass it on models that still accept it.
        if not self._anthropic_temperature_deprecated():
            stream_kwargs["temperature"] = self.config.temperature

        chunks: list[str] = []
        with self._client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        body = "".join(chunks)
        if use_prefill:
            # The model continues from the prefill '{' so prepend it back
            return "{" + body
        # No-prefill path: extract the first balanced JSON object from the response.
        return self._extract_json_object(body)

    # DEPRECATED: superseded by _chat_google_history; will be removed in next release
    def _chat_google(
        self, system_msg: str, user_msg: str, images: Optional[list] = None
    ) -> str:
        import PIL.Image

        # Rebuild model per-call so system_instruction is set correctly
        model = self._client.GenerativeModel(
            self.config.model,
            system_instruction=system_msg,
        )

        parts = [user_msg]
        if images:
            for img_path in images:
                parts.append(PIL.Image.open(str(img_path)))

        gen_config = {
            "temperature": self.config.temperature,
            "response_mime_type": "application/json",
        }
        if self.config.max_tokens is not None:
            gen_config["max_output_tokens"] = self.config.max_tokens

        resp = model.generate_content(
            parts,
            generation_config=gen_config,
        )
        return resp.text

    # ------------------------------------------------------------------
    # Provider-specific multi-turn history implementations
    # ------------------------------------------------------------------

    def _chat_openai_history(
        self,
        system: str,
        messages: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        tools: Optional[list[dict]] = None,
        tool_use_callback: Optional[Callable[[str, str], None]] = None,
    ):
        out_messages: list = [{"role": "system", "content": system}]
        for m in messages:
            converted = self._openai_convert_message(m)
            if isinstance(converted, list):
                out_messages.extend(converted)
            else:
                out_messages.append(converted)

        kwargs: dict = {
            "model": self.config.model,
            "messages": out_messages,
            "temperature": self.config.temperature,
        }
        # Newer models (o1, o3, o4, gpt-5.x) use max_completion_tokens
        if self._uses_completion_tokens():
            if self.config.max_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.max_tokens
            # gpt-5.x is a standard chat model — force JSON output
            # True reasoning models (o1/o3/o4) do not support response_format
            m_name = self.config.model.lower()
            _no_json_format = ("o1", "o3", "o4")
            if not any(m_name.startswith(p) for p in _no_json_format):
                # response_format=json_object is incompatible with tools; skip when tools active
                if tools is None:
                    kwargs["response_format"] = {"type": "json_object"}
        else:
            if self.config.max_tokens is not None:
                kwargs["max_tokens"] = self.config.max_tokens
            if tools is None:
                kwargs["response_format"] = {"type": "json_object"}

        # Activate DeepSeek thinking mode when configured.
        effort = getattr(self.config, "reasoning_effort", None)
        if effort and self.config.provider == "deepseek":
            kwargs["reasoning_effort"] = effort

        # Convert internal tool schemas → OpenAI function-calling shape
        if tools is not None:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object"}),
                    },
                }
                for t in tools
            ]

        # Non-streaming fast path — only when neither callback is set AND no tools
        if stream_callback is None and thinking_callback is None and tools is None:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        # Streaming path
        kwargs["stream"] = True
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        # Accumulator for tool calls keyed by index — stable index per call across deltas
        tool_acc: dict[int, dict] = {}
        finish_reason: Optional[str] = None

        stream = self._client.chat.completions.create(**kwargs)
        for event in stream:
            choices = getattr(event, "choices", None)
            if not choices:
                continue
            choice0 = choices[0]
            delta = getattr(choice0, "delta", None)
            fr = getattr(choice0, "finish_reason", None)
            if fr:
                finish_reason = fr
            if delta is None:
                continue
            # Reasoning content (DeepSeek thinking mode)
            rc = getattr(delta, "reasoning_content", None) or ""
            if rc:
                thinking_chunks.append(rc)
                if thinking_callback is not None:
                    try:
                        thinking_callback(rc)
                    except Exception as exc:
                        logger.warning("thinking_callback raised: %s", exc)
            # Tool-call deltas
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = getattr(tc, "index", 0) or 0
                slot = tool_acc.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    slot["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None)
                    if name:
                        slot["name"] = name
                    args_chunk = getattr(fn, "arguments", None)
                    if args_chunk:
                        slot["arguments"] += args_chunk
                        if tool_use_callback is not None and slot["name"]:
                            try:
                                tool_use_callback(slot["name"], args_chunk)
                            except Exception as exc:
                                logger.warning("tool_use_callback raised: %s", exc)
            # Final-answer text
            text = getattr(delta, "content", None) or ""
            if text:
                text_chunks.append(text)
                if stream_callback is not None:
                    try:
                        stream_callback(text)
                    except Exception as exc:
                        logger.warning("stream_callback raised: %s", exc)

        body = "".join(text_chunks)
        if tools is None:
            return body

        # Structured return when tools are active
        tool_calls: list[dict] = []
        for idx in sorted(tool_acc.keys()):
            slot = tool_acc[idx]
            args_str = slot["arguments"] or "{}"
            try:
                parsed_args = json.loads(args_str)
            except Exception:
                parsed_args = {"_raw_arguments": args_str}
            tool_calls.append({
                "id": slot["id"] or f"call_{idx}",
                "name": slot["name"] or "",
                "arguments": parsed_args,
            })
        if tool_calls:
            kind = "tool_use"
        elif body:
            kind = "text"
        else:
            kind = "stop"
        reasoning_full = "".join(thinking_chunks)
        return {
            "type": kind,
            "text": body,
            "tool_calls": tool_calls,
            "stop_reason": finish_reason or kind,
            "raw_response": body,
            "thinking_response": reasoning_full,
            # DeepSeek thinking-mode multi-turn requires the assistant's
            # `reasoning_content` to be passed back when the prior turn
            # involved a tool call. Surface it here so the agent loop can
            # round-trip it into the next request.
            "reasoning_content": reasoning_full or None,
        }

    # --- helpers for OpenAI message conversion ---

    def _openai_convert_message(self, m: dict):
        """Convert one internal-format message to OpenAI's wire shape.

        Internal messages may be:
          - {"role": "user"|"assistant", "content": <str>, "images": [...]}
          - {"role": "assistant", "content": [{"type":"text","text":...},
                                              {"type":"tool_use", id, name, input}]}
          - {"role": "user", "content": [{"type":"tool_result", tool_use_id, content}]}

        Returns either a single dict or a list of dicts (e.g., for assistant
        turns that mix text + tool_use, or user turns containing tool_result
        blocks which OpenAI requires as separate ``role="tool"`` messages).
        """
        role = m["role"]
        content = m.get("content")
        # Plain string content — keep simple shape
        if isinstance(content, str):
            content_parts = [{"type": "text", "text": content}]
            for img_path in m.get("images") or []:
                b64 = self._encode_image(img_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            out_msg: dict = {"role": role, "content": content_parts}
            # Preserve DeepSeek reasoning_content on plain-string assistant
            # turns too (defensive — current AgentLoop emits list-of-blocks).
            rc = m.get("reasoning_content")
            if role == "assistant" and rc and self.config.provider == "deepseek":
                out_msg["reasoning_content"] = rc
            return out_msg

        # List-of-blocks content (Anthropic-native shape)
        if isinstance(content, list):
            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    btype = blk.get("type")
                    if btype == "text":
                        text_parts.append(blk.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": blk.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": blk.get("name", ""),
                                "arguments": json.dumps(blk.get("input", {}), default=str),
                            },
                        })
                msg: dict = {"role": "assistant", "content": "".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                # DeepSeek thinking-mode multi-turn: when the prior assistant
                # turn carried a `reasoning_content` (chain-of-thought) AND
                # involved a tool call, the field MUST be passed back to the
                # API. OpenAI proper rejects this kwarg, so guard by provider.
                rc = m.get("reasoning_content")
                if rc and self.config.provider == "deepseek":
                    msg["reasoning_content"] = rc
                return msg
            if role == "user":
                # tool_result blocks become separate tool messages
                emitted: list[dict] = []
                text_parts: list[str] = []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    btype = blk.get("type")
                    if btype == "tool_result":
                        result_content = blk.get("content")
                        if isinstance(result_content, (dict, list)):
                            content_str = json.dumps(result_content, default=str)
                        else:
                            content_str = str(result_content)
                        emitted.append({
                            "role": "tool",
                            "tool_call_id": blk.get("tool_use_id", ""),
                            "content": content_str,
                        })
                    elif btype == "text":
                        text_parts.append(blk.get("text", ""))
                if text_parts:
                    emitted.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "\n".join(text_parts)}],
                    })
                return emitted or {"role": "user", "content": ""}
        # Fallback
        return {"role": role, "content": str(content)}

    def _chat_anthropic_history(
        self,
        system: str,
        messages: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        tools: Optional[list[dict]] = None,
        tool_use_callback: Optional[Callable[[str, str], None]] = None,
    ):
        out_messages: list = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                # Already in Anthropic content-block shape — pass through
                out_messages.append({"role": m["role"], "content": content})
                continue
            content_parts = [{"type": "text", "text": content}]
            for img_path in m.get("images") or []:
                b64 = self._encode_image(img_path)
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )
            out_messages.append({"role": m["role"], "content": content_parts})

        # Decide whether to enable extended thinking. When enabled:
        #   - prefill MUST NOT be present
        #   - temperature MUST NOT be set (Claude 4.x already rejects it)
        effort = getattr(self.config, "reasoning_effort", None)
        thinking_enabled = (
            effort is not None
            and effort in self._ANTHROPIC_EFFORT_BUDGET
            and self._anthropic_supports_thinking()
        )

        # Claude 4.x rejects assistant prefill; older models get a final '{' assistant
        # turn to coax JSON. With extended thinking on, prefill is also forbidden.
        # Tools-mode also forbids JSON prefill since the model needs to emit tool_use blocks.
        use_prefill = (
            (not self._anthropic_no_prefill())
            and (not thinking_enabled)
            and (tools is None)
        )
        if use_prefill:
            out_messages.append({"role": "assistant", "content": "{"})

        stream_kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 64000,
            "system": system,
            "messages": out_messages,
        }
        if not self._anthropic_temperature_deprecated() and not thinking_enabled:
            stream_kwargs["temperature"] = self.config.temperature
        if thinking_enabled:
            budget = self._ANTHROPIC_EFFORT_BUDGET[effort]
            stream_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Anthropic requires max_tokens >= budget_tokens + final_answer headroom.
            if stream_kwargs["max_tokens"] <= budget:
                stream_kwargs["max_tokens"] = budget + 16384

        # Anthropic-native tool schemas pass straight through.
        if tools is not None:
            stream_kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {"type": "object"}),
                }
                for t in tools
            ]

        chunks: list[str] = []
        thinking_chunks: list[str] = []
        # Tool-use accumulators keyed by content-block index
        tool_acc: dict[int, dict] = {}
        stop_reason: Optional[str] = None

        def _process_event(event):
            nonlocal stop_reason
            etype = getattr(event, "type", None)
            if etype == "content_block_start":
                block = getattr(event, "content_block", None)
                if block is not None and getattr(block, "type", None) == "tool_use":
                    idx = getattr(event, "index", len(tool_acc))
                    tool_acc[idx] = {
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "arguments": "",
                    }
            elif etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is None:
                    return
                dtype = getattr(delta, "type", None)
                idx = getattr(event, "index", 0)
                if dtype == "thinking_delta":
                    t = getattr(delta, "thinking", "") or ""
                    if t:
                        thinking_chunks.append(t)
                        if thinking_callback is not None:
                            try:
                                thinking_callback(t)
                            except Exception as exc:
                                logger.warning("thinking_callback raised: %s", exc)
                elif dtype == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    if text:
                        chunks.append(text)
                        if stream_callback is not None:
                            try:
                                stream_callback(text)
                            except Exception as exc:
                                logger.warning("stream_callback raised: %s", exc)
                elif dtype == "input_json_delta":
                    partial = getattr(delta, "partial_json", "") or ""
                    if partial and idx in tool_acc:
                        tool_acc[idx]["arguments"] += partial
                        if tool_use_callback is not None:
                            try:
                                tool_use_callback(
                                    tool_acc[idx].get("name") or "",
                                    partial,
                                )
                            except Exception as exc:
                                logger.warning("tool_use_callback raised: %s", exc)
            elif etype == "message_delta":
                delta = getattr(event, "delta", None)
                if delta is not None:
                    sr = getattr(delta, "stop_reason", None)
                    if sr:
                        stop_reason = sr

        try:
            with self._client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    _process_event(event)
        except TypeError as exc:
            # Older SDK versions may reject the `thinking` kwarg or yield events
            # in an incompatible shape. Fall back to text_stream-only path.
            if thinking_enabled and tools is None:
                logger.warning(
                    "Anthropic SDK does not support extended-thinking event "
                    "iteration in this version (%s); falling back to text_stream.",
                    exc,
                )
                stream_kwargs.pop("thinking", None)
                chunks.clear()
                with self._client.messages.stream(**stream_kwargs) as stream:
                    for text in stream.text_stream:
                        chunks.append(text)
                        if stream_callback is not None:
                            try:
                                stream_callback(text)
                            except Exception as cb_exc:
                                logger.warning("stream_callback raised: %s", cb_exc)
            else:
                raise

        body = "".join(chunks)
        if tools is None:
            if use_prefill:
                return "{" + body
            return self._extract_json_object(body)

        # Structured return for tool-use mode
        tool_calls: list[dict] = []
        for idx in sorted(tool_acc.keys()):
            slot = tool_acc[idx]
            args_str = slot["arguments"] or "{}"
            try:
                parsed_args = json.loads(args_str)
            except Exception:
                parsed_args = {"_raw_arguments": args_str}
            tool_calls.append({
                "id": slot["id"] or f"toolu_{idx}",
                "name": slot["name"] or "",
                "arguments": parsed_args,
            })
        if tool_calls:
            kind = "tool_use"
        elif body:
            kind = "text"
        else:
            kind = "stop"
        reasoning_full = "".join(thinking_chunks)
        return {
            "type": kind,
            "text": body,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason or kind,
            "raw_response": body,
            "thinking_response": reasoning_full,
            # Anthropic handles thinking signatures via the SDK on its own
            # round-trip path, so this is informational here. We surface it
            # for parity with the DeepSeek path (and per-turn artifact
            # transparency).
            "reasoning_content": reasoning_full or None,
        }

    def _chat_google_history(
        self,
        system: str,
        messages: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        import PIL.Image

        if not messages or messages[-1].get("role") != "user":
            raise ValueError(
                "Google provider requires the final history turn to be a user message"
            )

        model = self._client.GenerativeModel(
            self.config.model,
            system_instruction=system,
        )

        history = []
        for m in messages[:-1]:
            role = "model" if m["role"] == "assistant" else m["role"]
            history.append({"role": role, "parts": [m["content"]]})

        last = messages[-1]
        parts: list = [last["content"]]
        for img_path in last.get("images") or []:
            parts.append(PIL.Image.open(str(img_path)))

        gen_config = {
            "temperature": self.config.temperature,
            "response_mime_type": "application/json",
        }
        if self.config.max_tokens is not None:
            gen_config["max_output_tokens"] = self.config.max_tokens

        chat = model.start_chat(history=history)
        if stream_callback is None:
            resp = chat.send_message(parts, generation_config=gen_config)
            return resp.text

        # Streaming path — Google SDK supports stream=True on send_message,
        # which returns an iterable of chunks each exposing a `.text` field.
        chunks: list[str] = []
        resp = chat.send_message(parts, generation_config=gen_config, stream=True)
        for chunk in resp:
            text = getattr(chunk, "text", None) or ""
            if text:
                chunks.append(text)
                try:
                    stream_callback(text)
                except Exception as exc:
                    logger.warning("stream_callback raised: %s", exc)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Google/Gemini tool-use (function calling)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_gemini_schema(schema: dict):
        """Reduce an internal JSON-schema to the OpenAPI subset Gemini accepts.
        Drops unsupported keys (additionalProperties, default, ...) and returns
        None for an empty object (a no-argument function)."""
        if not isinstance(schema, dict):
            return None
        allowed = {"type", "description", "properties", "required", "items", "enum", "nullable"}
        _type_map = {"object": "OBJECT", "string": "STRING", "integer": "INTEGER",
                     "number": "NUMBER", "boolean": "BOOLEAN", "array": "ARRAY"}
        out = {k: v for k, v in schema.items() if k in allowed}
        if isinstance(out.get("type"), str):
            out["type"] = _type_map.get(out["type"].lower(), out["type"].upper())
        props = out.get("properties")
        if isinstance(props, dict):
            cleaned = {}
            for name, sub in props.items():
                c = LLMClient._sanitize_gemini_schema(sub)
                cleaned[name] = c if c is not None else {"type": "STRING"}
            out["properties"] = cleaned
        if isinstance(out.get("items"), dict):
            out["items"] = LLMClient._sanitize_gemini_schema(out["items"]) or {"type": "STRING"}
        # A no-property object is not a valid Gemini parameter schema.
        if out.get("type") in (None, "OBJECT") and not out.get("properties"):
            return None
        out.setdefault("type", "OBJECT")
        return out

    def _chat_google_history_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
        tool_use_callback: Optional[Callable[[str, str], None]] = None,
    ) -> dict:
        """Gemini function-calling bridge for the AgentLoop.

        Translates the loop's Anthropic-native content blocks into Gemini
        Contents, runs one generation with function declarations, and returns
        the loop's expected dict shape ({text, tool_calls, stop_reason, ...}).
        """
        from google.generativeai import protos

        def _struct_to_dict(args) -> dict:
            try:
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(args) if args is not None else {}
            except Exception:
                try:
                    return dict(args)
                except Exception:
                    return {}

        # Function declarations
        fdecls = []
        for t in tools:
            params = self._sanitize_gemini_schema(t.get("input_schema") or {})
            kwargs = {"name": t["name"], "description": (t.get("description") or "")[:1024]}
            if params is not None:
                kwargs["parameters"] = params
            fdecls.append(protos.FunctionDeclaration(**kwargs))
        gtool = protos.Tool(function_declarations=fdecls)

        model = self._client.GenerativeModel(
            self.config.model, system_instruction=system, tools=[gtool],
        )

        # id -> function name map from prior assistant tool_use blocks
        id2name: dict = {}
        for m in messages:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "tool_use":
                        id2name[b.get("id", "")] = b.get("name", "")

        # Gemini 3.x requires a `thought_signature` on every function_call part
        # replayed in history, but the legacy google.generativeai SDK's proto
        # cannot carry that field. We therefore flatten prior tool calls and
        # their results into TEXT (never sending function_call/function_response
        # parts back). The model still sees the full interaction and issues
        # fresh function calls on the current turn (which we parse below).
        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            content = m.get("content")
            texts = []
            if isinstance(content, str):
                if content:
                    texts.append(content)
            elif isinstance(content, list):
                for b in content:
                    bt = b.get("type")
                    if bt == "text" and b.get("text"):
                        texts.append(b["text"])
                    elif bt == "tool_use":
                        args = json.dumps(b.get("input") or {}, default=str)
                        texts.append(f"[Called tool `{b.get('name','')}` with arguments: {args}]")
                    elif bt == "tool_result":
                        name = id2name.get(b.get("tool_use_id", ""), "tool")
                        raw = b.get("content")
                        raw_str = raw if isinstance(raw, str) else json.dumps(raw, default=str)
                        texts.append(f"[Result of tool `{name}`]:\n{raw_str}")
            joined = "\n\n".join(t for t in texts if t)
            if joined:
                contents.append(protos.Content(role=role, parts=[protos.Part(text=joined)]))

        gen_config = {"temperature": self.config.temperature}
        if self.config.max_tokens is not None:
            gen_config["max_output_tokens"] = self.config.max_tokens

        resp = model.generate_content(contents, generation_config=gen_config)

        text_out: list = []
        tool_calls: list = []
        finish = ""
        try:
            cand = resp.candidates[0]
            finish = str(getattr(cand, "finish_reason", ""))
            for part in cand.content.parts:
                if getattr(part, "text", ""):
                    text_out.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", ""):
                    args = _struct_to_dict(getattr(fc, "args", None))
                    call = {"id": f"{fc.name}_{len(tool_calls)}", "name": fc.name, "arguments": args}
                    tool_calls.append(call)
                    if tool_use_callback is not None:
                        try:
                            tool_use_callback(fc.name, json.dumps(args, default=str))
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Gemini tool-response parse failed: %s", exc)

        text = "".join(text_out)
        if stream_callback is not None and text:
            try:
                stream_callback(text)
            except Exception:
                pass

        return {
            "type": "tool_use" if tool_calls else ("text" if text else "stop"),
            "text": text,
            "tool_calls": tool_calls,
            "stop_reason": finish or ("tool_use" if tool_calls else "stop"),
            "raw_response": text,
            "thinking_response": "",
            "reasoning_content": None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        with open(str(path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
