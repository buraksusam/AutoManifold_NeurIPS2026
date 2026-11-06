import json
from unittest.mock import MagicMock
import pytest


def _make_llm(response: str):
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


def test_dynamic_agent_uses_custom_system_message():
    from pipeline.agents.dynamic_agent import DynamicAgent
    llm = _make_llm('{"result": "done"}')
    agent = DynamicAgent(llm, "TestRole", "My custom system msg", "My task prompt")
    agent.run({})
    call_kwargs = llm.chat.call_args[1]
    assert call_kwargs["system_message"] == "My custom system msg"


def test_dynamic_agent_uses_task_prompt():
    from pipeline.agents.dynamic_agent import DynamicAgent
    llm = _make_llm('{"result": "done"}')
    agent = DynamicAgent(llm, "Analyzer", "sys", "do this analysis")
    agent.run({})
    call_kwargs = llm.chat.call_args[1]
    assert call_kwargs["user_message"] == "do this analysis"


def test_dynamic_agent_returns_parsed_dict():
    from pipeline.agents.dynamic_agent import DynamicAgent
    llm = _make_llm('{"key": "value", "score": 42}')
    agent = DynamicAgent(llm, "R", "s", "p")
    result = agent.run({})
    assert result == {"key": "value", "score": 42}


def test_dynamic_agent_sets_agent_name_from_role():
    from pipeline.agents.dynamic_agent import DynamicAgent
    llm = _make_llm('{}')
    agent = DynamicAgent(llm, "ManifoldAnalyzer", "s", "p")
    assert "ManifoldAnalyzer" in agent.AGENT_NAME
