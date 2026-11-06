"""Tests for LLMClient.chat_history multi-turn support."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import LLMConfig
from pipeline.llm_client import LLMClient


def _make_client(provider: str, model: str = "test-model") -> LLMClient:
    cfg = LLMConfig(provider=provider, model=model, api_key="x", temperature=0.0)
    # Bypass real SDK init
    with patch.object(LLMClient, "_init_client"):
        c = LLMClient(cfg)
    return c


def test_chat_history_openai_message_shape():
    c = _make_client("openai")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": 1}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    c._client = fake_client

    out = c.chat_history(
        system="SYS",
        messages=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
    )
    assert out == '{"ok": 1}'
    sent = fake_client.chat.completions.create.call_args.kwargs
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    # User content gets wrapped in [{"type": "text", "text": ...}] by our shape;
    # we only assert the role + extract the text.
    def _text(m):
        c = m["content"]
        if isinstance(c, str):
            return c
        return c[0]["text"]
    assert sent["messages"][1]["role"] == "user" and _text(sent["messages"][1]) == "u1"
    assert sent["messages"][2]["role"] == "assistant" and _text(sent["messages"][2]) == "a1"
    assert sent["messages"][3]["role"] == "user" and _text(sent["messages"][3]) == "u2"


def _make_text_event(text: str):
    """Build a fake content_block_delta event carrying a text_delta."""
    delta = MagicMock(type="text_delta", text=text)
    return MagicMock(type="content_block_delta", delta=delta)


def _make_thinking_event(text: str):
    """Build a fake content_block_delta event carrying a thinking_delta."""
    delta = MagicMock(type="thinking_delta", thinking=text)
    return MagicMock(type="content_block_delta", delta=delta)


class _FakeAnthropicStream:
    """Context-manager iterable mimicking anthropic.MessageStream."""
    def __init__(self, events):
        self._events = events
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter(self._events)
    @property
    def text_stream(self):
        # Fallback: yield any text-delta events' .text payloads.
        return iter(
            getattr(e.delta, "text", "")
            for e in self._events
            if getattr(e.delta, "type", None) == "text_delta"
        )


def test_chat_history_anthropic_message_shape():
    c = _make_client("anthropic", model="claude-opus-4-7")

    captured_kwargs = {}

    def stream_fn(**kw):
        captured_kwargs.update(kw)
        return _FakeAnthropicStream([_make_text_event('"ok": 1}')])

    fake_messages = MagicMock()
    fake_messages.stream = stream_fn
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    c._client = fake_client

    out = c.chat_history(
        system="SYS",
        messages=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
    )
    # Claude 4.x = no prefill path: returned text should round-trip the streamed body
    assert "ok" in out
    assert captured_kwargs["system"] == "SYS"
    msgs = captured_kwargs["messages"]
    # First three turns are passed through as-is; no prefill on Claude 4.x
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["text"] == "u1"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["text"] == "a1"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["text"] == "u2"


def test_chat_history_openai_streaming_invokes_callback():
    """When stream_callback is provided, OpenAI path streams via SDK and relays each delta."""
    c = _make_client("openai")

    def _make_chunk(text: str | None):
        delta = MagicMock(content=text)
        choice = MagicMock(delta=delta)
        ev = MagicMock(choices=[choice])
        return ev

    deltas = ["{\"a", "ns\": ", "1}"]

    def _create(**kw):
        # Streaming flag must have been set by the client
        assert kw.get("stream") is True
        return iter([_make_chunk(d) for d in deltas])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    c._client = fake_client

    received: list[str] = []
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "u1"}],
        stream_callback=lambda chunk: received.append(chunk),
    )
    assert received == deltas
    assert out == "".join(deltas)


def test_chat_history_anthropic_streaming_invokes_callback():
    """Anthropic stream relays each text delta to the callback."""
    c = _make_client("anthropic", model="claude-opus-4-7")

    deltas = ["foo", "bar", "baz"]
    events = [_make_text_event(d) for d in deltas]

    fake_messages = MagicMock()
    fake_messages.stream = lambda **kw: _FakeAnthropicStream(events)
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    c._client = fake_client

    received: list[str] = []
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "u1"}],
        stream_callback=lambda chunk: received.append(chunk),
    )
    assert received == deltas
    # Claude 4.x no-prefill path strips back to the JSON-extracted body;
    # since we sent non-JSON strings, _extract_json_object falls through.
    assert out == "foobarbaz"


def test_chat_history_openai_no_streaming_when_callback_absent():
    """Without a callback, the OpenAI path must NOT set stream=True."""
    c = _make_client("openai")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": 1}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    c._client = fake_client

    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "u1"}],
    )
    assert out == '{"ok": 1}'
    sent = fake_client.chat.completions.create.call_args.kwargs
    # Must not have switched into streaming mode
    assert "stream" not in sent or sent["stream"] is False


def test_chat_history_deepseek_streams_reasoning_content():
    """DeepSeek thinking-mode streams `delta.reasoning_content` BEFORE `delta.content`.
    Reasoning chunks must route to thinking_callback; content chunks to stream_callback;
    the final return is content-only concatenation."""
    c = _make_client("deepseek", model="deepseek-v4-pro")
    c.config.reasoning_effort = "high"

    def _make_chunk(reasoning: str | None = None, content: str | None = None):
        delta = MagicMock(reasoning_content=reasoning, content=content)
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice])

    # Reasoning comes first, then final answer
    events = [
        _make_chunk(reasoning="thinking step 1..."),
        _make_chunk(reasoning="thinking step 2."),
        _make_chunk(content='{"answer'),
        _make_chunk(content='": 42}'),
    ]

    captured_kwargs = {}

    def _create(**kw):
        captured_kwargs.update(kw)
        assert kw.get("stream") is True
        return iter(events)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    c._client = fake_client

    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "u1"}],
        stream_callback=lambda x: answer_chunks.append(x),
        thinking_callback=lambda x: thinking_chunks.append(x),
    )

    # Reasoning routed to thinking_callback in arrival order
    assert thinking_chunks == ["thinking step 1...", "thinking step 2."]
    # Content routed to stream_callback in arrival order
    assert answer_chunks == ['{"answer', '": 42}']
    # Return is content-only
    assert out == '{"answer": 42}'
    # reasoning_effort kwarg was passed
    assert captured_kwargs.get("reasoning_effort") == "high"


def test_chat_history_anthropic_streams_thinking_delta():
    """Anthropic extended thinking yields thinking_delta events BEFORE text_delta.
    Each must route to its respective callback; final return is text_delta-only."""
    c = _make_client("anthropic", model="claude-opus-4-7")
    c.config.reasoning_effort = "high"

    events = [
        _make_thinking_event("Let me think... "),
        _make_thinking_event("the answer is 42."),
        _make_text_event('{"answer'),
        _make_text_event('": 42}'),
    ]

    captured_kwargs = {}

    def stream_fn(**kw):
        captured_kwargs.update(kw)
        return _FakeAnthropicStream(events)

    fake_messages = MagicMock()
    fake_messages.stream = stream_fn
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    c._client = fake_client

    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "u1"}],
        stream_callback=lambda x: answer_chunks.append(x),
        thinking_callback=lambda x: thinking_chunks.append(x),
    )

    assert thinking_chunks == ["Let me think... ", "the answer is 42."]
    assert answer_chunks == ['{"answer', '": 42}']
    # Final return contains only the text deltas (after JSON extraction)
    assert out == '{"answer": 42}'
    # `thinking` parameter was passed with the right shape
    th = captured_kwargs.get("thinking")
    assert th == {"type": "enabled", "budget_tokens": 16384}
    # No prefill assistant turn was appended
    msgs = captured_kwargs["messages"]
    assert all(m.get("role") != "assistant" or m["content"] != "{" for m in msgs)
    # No temperature when extended thinking is on
    assert "temperature" not in captured_kwargs


def test_chat_backward_compat_calls_chat_history():
    """Existing single-turn chat() must still work — it now wraps chat_history."""
    c = _make_client("openai")
    with patch.object(c, "chat_history", return_value='{"ok": 1}') as mocked:
        out = c.chat(system_message="SYS", user_message="hello")
        assert out == '{"ok": 1}'
        sent = mocked.call_args.kwargs
        assert sent["system"] == "SYS"
        assert sent["messages"] == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Tool-use tests
# ---------------------------------------------------------------------------


def _make_anthropic_tool_use_start(idx: int, tool_id: str, name: str):
    block = MagicMock(type="tool_use", id=tool_id)
    block.name = name  # `name` is reserved on MagicMock — assign explicitly
    return MagicMock(type="content_block_start", index=idx, content_block=block)


def _make_anthropic_input_json_delta(idx: int, partial: str):
    delta = MagicMock(type="input_json_delta", partial_json=partial)
    return MagicMock(type="content_block_delta", index=idx, delta=delta)


def _make_anthropic_message_delta(stop_reason: str):
    delta = MagicMock(stop_reason=stop_reason)
    return MagicMock(type="message_delta", delta=delta)


def test_chat_history_anthropic_tool_use():
    c = _make_client("anthropic", model="claude-opus-4-7")
    events = [
        _make_anthropic_tool_use_start(0, "toolu_abc", "compute_pca_summary"),
        _make_anthropic_input_json_delta(0, '{"x":'),
        _make_anthropic_input_json_delta(0, ' 1}'),
        _make_anthropic_message_delta("tool_use"),
    ]
    fake_messages = MagicMock()
    fake_messages.stream = lambda **kw: _FakeAnthropicStream(events)
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    c._client = fake_client

    tools = [{
        "name": "compute_pca_summary",
        "description": "PCA",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }]
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "go"}],
        tools=tools,
    )
    assert isinstance(out, dict)
    assert out["type"] == "tool_use"
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["name"] == "compute_pca_summary"
    assert tc["id"] == "toolu_abc"
    assert tc["arguments"] == {"x": 1}
    assert out["stop_reason"] == "tool_use"


def test_chat_history_openai_tool_use():
    c = _make_client("openai")

    def _make_chunk(tool_idx: int | None = None, tc_id: str | None = None,
                    name: str | None = None, args: str | None = None,
                    finish_reason: str | None = None):
        if tool_idx is None:
            choice = MagicMock(delta=MagicMock(content=None, tool_calls=None,
                                               reasoning_content=None),
                               finish_reason=finish_reason)
            return MagicMock(choices=[choice])
        fn = MagicMock(name=name, arguments=args)
        # MagicMock attributes from kwargs work, but `name` is reserved on MagicMock:
        fn.name = name
        fn.arguments = args
        tc = MagicMock(index=tool_idx, id=tc_id, function=fn)
        delta = MagicMock(content=None, reasoning_content=None, tool_calls=[tc])
        choice = MagicMock(delta=delta, finish_reason=finish_reason)
        return MagicMock(choices=[choice])

    events = [
        _make_chunk(tool_idx=0, tc_id="call_xyz", name="compute_pca_summary", args='{"x":'),
        _make_chunk(tool_idx=0, args=' 2}'),
        _make_chunk(finish_reason="tool_calls"),
    ]

    def _create(**kw):
        assert kw.get("stream") is True
        assert "tools" in kw
        # tools converted into OpenAI shape
        assert kw["tools"][0]["type"] == "function"
        assert kw["tools"][0]["function"]["name"] == "compute_pca_summary"
        return iter(events)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    c._client = fake_client

    tools = [{
        "name": "compute_pca_summary",
        "description": "PCA",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }]
    out = c.chat_history(
        system="SYS",
        messages=[{"role": "user", "content": "go"}],
        tools=tools,
    )
    assert isinstance(out, dict)
    assert out["type"] == "tool_use"
    assert out["tool_calls"][0]["name"] == "compute_pca_summary"
    assert out["tool_calls"][0]["arguments"] == {"x": 2}
    assert out["tool_calls"][0]["id"] == "call_xyz"


def test_chat_history_deepseek_outgoing_includes_reasoning_content():
    """When provider=='deepseek' and a prior assistant turn carried
    `reasoning_content`, the outgoing wire message MUST include it as a
    top-level field on the assistant message. DeepSeek's thinking-mode API
    rejects (400) requests where this is missing after a tool call.
    """
    c = _make_client("deepseek", model="deepseek-v4-pro")

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": 1}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    c._client = fake_client

    # Internal message shape produced by AgentLoop._append_assistant_turn:
    # list-of-blocks content with reasoning_content sibling field.
    assistant_with_reasoning = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "c1", "name": "echo", "input": {}},
        ],
        "reasoning_content": "thinking step 1",
    }
    tool_result_user = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "{}"},
        ],
    }

    c.chat_history(
        system="SYS",
        messages=[
            {"role": "user", "content": "u1"},
            assistant_with_reasoning,
            tool_result_user,
        ],
    )

    sent = fake_client.chat.completions.create.call_args.kwargs
    msgs = sent["messages"]
    asst = [m for m in msgs if m.get("role") == "assistant"]
    assert asst, "expected an assistant message in outgoing payload"
    assert asst[0].get("reasoning_content") == "thinking step 1", (
        "DeepSeek requires `reasoning_content` to be passed back when a "
        "tool call is in the history; outgoing assistant message dropped it."
    )


def test_chat_history_openai_outgoing_omits_reasoning_content():
    """OpenAI proper rejects `reasoning_content` on the wire. Even if our
    internal message carries it (e.g., from a config swap), we must NOT emit
    it for provider=='openai'.
    """
    c = _make_client("openai")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": 1}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    c._client = fake_client

    c.chat_history(
        system="SYS",
        messages=[
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "reasoning_content": "should not be on the wire",
            },
            {"role": "user", "content": "u2"},
        ],
    )
    sent = fake_client.chat.completions.create.call_args.kwargs
    msgs = sent["messages"]
    asst = [m for m in msgs if m.get("role") == "assistant"]
    assert asst
    assert "reasoning_content" not in asst[0]


def test_chat_history_no_tools_returns_string():
    """Backwards compat: when tools=None, return type stays str."""
    c = _make_client("openai")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": 1}'))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    c._client = fake_client

    out = c.chat_history(system="SYS", messages=[{"role": "user", "content": "u1"}])
    assert isinstance(out, str)
    assert out == '{"ok": 1}'
