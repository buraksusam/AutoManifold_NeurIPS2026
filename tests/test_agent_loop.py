"""Tests for AgentLoop — drive a mock LLM through the loop."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np

from pipeline.agent_loop import AgentLoop, SessionState
from pipeline.tools import ToolRegistry, register_finalize_tool


class _MockToolUseLLM:
    """Mock LLMClient.chat_history that returns a scripted sequence of responses."""

    def __init__(self, scripted: List[Dict[str, Any]]):
        self._scripted = scripted
        self._idx = 0
        self.calls: List[Dict[str, Any]] = []

    def chat_history(self, system, messages, tools=None, stream_callback=None,
                     thinking_callback=None, tool_use_callback=None, max_retries=3):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._idx >= len(self._scripted):
            # Default to a non-tool-call text response after script exhausts
            return {
                "type": "text",
                "text": "(no further actions)",
                "tool_calls": [],
                "stop_reason": "end_turn",
                "raw_response": "",
                "thinking_response": "",
            }
        resp = self._scripted[self._idx]
        self._idx += 1
        return resp


def _make_minimal_state(tmp_dir: str) -> SessionState:
    return SessionState(
        data=np.zeros((4, 3)),
        labels=None,
        label_names=None,
        primary_goal="local",
        output_dir=tmp_dir,
        dr_engine=object(),
        metrics_engine=object(),
        clustering_engine=object(),
    )


def _tool_use_resp(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "text": "",
        "tool_calls": tool_calls,
        "stop_reason": "tool_use",
        "raw_response": "",
        "thinking_response": "",
    }


def _text_resp(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "tool_calls": [],
        "stop_reason": "end_turn",
        "raw_response": text,
        "thinking_response": "",
    }


def test_agent_loop_runs_until_finalize():
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)

    registry = ToolRegistry()
    dispatched: List[str] = []

    def mk_handler(name: str):
        def handler(args, st):
            dispatched.append(name)
            return {"tool": name, "args": args}
        return handler

    for n in ["compute_baseline_calibration", "analyze-data",
              "validate_custom_code", "synthesize-algorithm"]:
        registry.register(
            n, {"description": n,
                "input_schema": {"type": "object", "properties": {}, "required": []}},
            mk_handler(n),
        )
    register_finalize_tool(registry)

    scripted = [
        _tool_use_resp([{"id": "c1", "name": "compute_baseline_calibration", "arguments": {}}]),
        _tool_use_resp([{"id": "c2", "name": "analyze-data", "arguments": {"task_context": "x"}}]),
        _tool_use_resp([{"id": "c3", "name": "synthesize-algorithm", "arguments": {"task_context": "y"}}]),
        _tool_use_resp([{"id": "c4", "name": "validate_custom_code", "arguments": {"code": "x"}}]),
        _tool_use_resp([{"id": "c5", "name": "finalize_design",
                         "arguments": {"final_report": {"algorithm_name": "Foo"}}}]),
    ]
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="PLANNER",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.add_initial_user_message("hi")
    final = loop.run()
    assert state.done is True
    assert final.get("algorithm_name") == "Foo"
    # All scripted tools dispatched in order (excluding finalize, which is its own tool)
    assert dispatched == [
        "compute_baseline_calibration",
        "analyze-data",
        "synthesize-algorithm",
        "validate_custom_code",
    ]
    # Per-iteration artifacts written
    turns_dir = os.path.join(tmp, "agent_turns")
    files = sorted(os.listdir(turns_dir))
    assert len(files) == 5
    # Sanity-check first artifact shape
    with open(os.path.join(turns_dir, files[0])) as f:
        payload = json.load(f)
    assert payload["iteration_index"] == 1
    assert payload["tool_calls"][0]["name"] == "compute_baseline_calibration"


def test_agent_loop_max_iterations_cap():
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()

    def echo(args, st):
        return {"echoed": True}

    registry.register(
        "echo",
        {"description": "echo",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        echo,
    )

    # Build a script that always calls 'echo' — never finalize.
    scripted = [_tool_use_resp([{"id": f"c{i}", "name": "echo", "arguments": {}}]) for i in range(10)]
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.MAX_ITERATIONS = 3   # tighten so the test runs fast
    loop.add_initial_user_message("hi")
    final = loop.run()
    assert state.done is False
    assert "warning" in final
    assert "max iterations" in final["warning"]
    assert loop.iteration_count == 3


def test_agent_loop_text_only_response_nudges():
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()
    register_finalize_tool(registry)

    scripted = [
        _text_resp("I'm thinking..."),  # Triggers nudge #1
        _text_resp("Still thinking..."),  # Triggers nudge #2
        _tool_use_resp([{"id": "f", "name": "finalize_design",
                         "arguments": {"final_report": {"algorithm_name": "Z"}}}]),
    ]
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.add_initial_user_message("hi")
    final = loop.run()
    assert state.done is True
    assert final["algorithm_name"] == "Z"
    # A successful tool call resets the nudge counter, so the live count is 0.
    # Verify the two nudges actually fired by counting nudge user-messages.
    nudge_msgs = [
        m for m in loop._messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "no tool call" in m["content"].lower()
    ]
    assert len(nudge_msgs) == 2


def test_agent_loop_skill_output_grace_does_not_count_as_nudge():
    """A text-only turn immediately following a skill tool call is the skill's
    JSON output and must NOT consume nudge budget."""
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()
    register_finalize_tool(registry)

    def skill_handler(args, st):
        return {"skill_name": "design-affinity", "skill_body": "..."}
    registry.register(
        "design-affinity",
        {"name": "design-affinity", "description": "x", "input_schema": {"type": "object"}},
        skill_handler,
    )

    scripted = [
        _tool_use_resp([{"id": "s1", "name": "design-affinity", "arguments": {}}]),
        _text_resp('{"affinity": "..."}'),  # skill JSON output — must be graced
        _text_resp("more prose"),            # this one IS a real nudge (#1)
        _tool_use_resp([{"id": "f", "name": "finalize_design",
                         "arguments": {"final_report": {"algorithm_name": "Z"}}}]),
    ]
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        skill_names=["design-affinity"],
        verbose=False,
    )
    loop.add_initial_user_message("hi")
    final = loop.run()
    assert state.done is True
    assert final["algorithm_name"] == "Z"
    nudge_msgs = [
        m for m in loop._messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "no tool call" in m["content"].lower()
    ]
    assert len(nudge_msgs) == 1
    grace_msgs = [
        m for m in loop._messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "skill output received" in m["content"].lower()
    ]
    assert len(grace_msgs) == 1


def test_agent_loop_text_only_terminates_after_max_nudges():
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()
    scripted = [_text_resp("hmm")] * 10
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.MAX_NUDGES = 2
    loop.add_initial_user_message("hi")
    final = loop.run()
    # Should bail after MAX_NUDGES nudges
    assert state.done is False


def test_agent_loop_tool_call_callback_fires_call_and_result():
    """tool_call_callback receives ('call', payload) then ('result', payload)
    per dispatched tool, with iteration index."""
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()

    def baseline_handler(args, st):
        return {"winner": "umap", "winner_score": 7.4, "intent": "global", "extra": {"deep": "x" * 5000}}
    registry.register(
        "compute_baseline_calibration",
        {"description": "baseline",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        baseline_handler,
    )
    register_finalize_tool(registry)

    scripted = [
        _tool_use_resp([{"id": "c1", "name": "compute_baseline_calibration", "arguments": {}}]),
        _tool_use_resp([{"id": "c2", "name": "finalize_design",
                         "arguments": {"final_report": {"algorithm_name": "Bar"}}}]),
    ]
    llm = _MockToolUseLLM(scripted)

    events: List[Dict[str, Any]] = []

    def tool_cb(event_type, payload):
        events.append({"event": event_type, **payload})

    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        tool_call_callback=tool_cb,
        verbose=False,
    )
    loop.add_initial_user_message("hi")
    loop.run()

    # Should have 4 events: call+result for baseline, call+result for finalize
    assert len(events) == 4
    assert events[0]["event"] == "call"
    assert events[0]["tool_name"] == "compute_baseline_calibration"
    assert events[0]["iteration"] == 1
    assert events[1]["event"] == "result"
    assert events[1]["tool_name"] == "compute_baseline_calibration"
    # baseline summary keeps winner/score, drops the deep blob
    summary = events[1]["result_summary"]
    assert summary["winner"] == "umap"
    assert summary["winner_score"] == 7.4
    assert "extra" not in summary
    assert events[2]["event"] == "call"
    assert events[2]["tool_name"] == "finalize_design"
    assert events[3]["event"] == "result"
    assert events[3]["tool_name"] == "finalize_design"


def test_agent_loop_preserves_deepseek_reasoning_content_across_iterations():
    """Round-tripping reasoning_content from one iteration to the next is
    required by DeepSeek thinking mode whenever a tool call is involved.

    Asserts:
      1. The appended assistant turn carries `reasoning_content` as a sibling
         field on the message dict.
      2. The next call to `chat_history` (iter 2) receives that assistant
         message with `reasoning_content` preserved.
    """
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()

    def echo_handler(args, st):
        return {"echoed": True}

    registry.register(
        "echo",
        {"description": "echo",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        echo_handler,
    )
    register_finalize_tool(registry)

    # Iter 1: tool_use with reasoning_content. Iter 2: finalize.
    iter1 = {
        "type": "tool_use",
        "text": "",
        "tool_calls": [{"id": "c1", "name": "echo", "arguments": {}}],
        "stop_reason": "tool_use",
        "raw_response": "",
        "thinking_response": "thinking step 1",
        "reasoning_content": "thinking step 1",
    }
    iter2 = _tool_use_resp([{"id": "f", "name": "finalize_design",
                             "arguments": {"final_report": {"algorithm_name": "Q"}}}])
    llm = _MockToolUseLLM([iter1, iter2])

    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="P",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.add_initial_user_message("hi")
    loop.run()

    # 1. The assistant turn appended after iter1 must carry reasoning_content.
    assistant_turns = [m for m in loop._messages if m.get("role") == "assistant"]
    assert assistant_turns, "expected at least one assistant turn in messages"
    assert assistant_turns[0].get("reasoning_content") == "thinking step 1", (
        "assistant turn lost reasoning_content (DeepSeek would 400 on next request)"
    )

    # 2. The messages snapshot passed to chat_history on iter2 must contain
    # that assistant turn with reasoning_content preserved.
    assert len(llm.calls) >= 2
    iter2_messages = llm.calls[1]["messages"]
    iter2_assistant = [m for m in iter2_messages if m.get("role") == "assistant"]
    assert iter2_assistant, "iter2 should have seen the prior assistant turn"
    assert iter2_assistant[0].get("reasoning_content") == "thinking step 1"


def test_agent_loop_passes_initial_user_message_to_llm():
    tmp = tempfile.mkdtemp()
    state = _make_minimal_state(tmp)
    registry = ToolRegistry()
    register_finalize_tool(registry)

    scripted = [
        _tool_use_resp([{"id": "f", "name": "finalize_design",
                         "arguments": {"final_report": {}}}]),
    ]
    llm = _MockToolUseLLM(scripted)
    loop = AgentLoop(
        llm_client=llm,
        planner_prompt="MY_PLANNER",
        tool_registry=registry,
        session_state=state,
        verbose=False,
    )
    loop.add_initial_user_message("HELLO_INITIAL")
    loop.run()
    first_call = llm.calls[0]
    assert first_call["messages"][0]["role"] == "user"
    assert first_call["messages"][0]["content"] == "HELLO_INITIAL"
    # Tool schemas were forwarded
    assert any(s["name"] == "finalize_design" for s in first_call["tools"])
