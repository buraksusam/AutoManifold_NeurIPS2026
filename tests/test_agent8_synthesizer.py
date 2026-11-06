import json
from unittest.mock import MagicMock

import pytest


def _make_llm(response: str):
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


_SYNTH_RESPONSE = json.dumps({
    "algorithm_specification": {
        "name": "TestAlgo",
        "summary": "A test algorithm",
    },
    "custom_embedding_code": "def custom_embedding(data, labels, params):\n    import numpy as np\n    from sklearn.decomposition import PCA\n    return PCA(n_components=2).fit_transform(data)",
    "default_params": {"n_components": 2},
    "param_descriptions": {"n_components": "output dimensions"},
})

_AGENT_CTX = {
    "agent1_output": {"summary": "data"},
    "agent2_output": {},
    "agent3_output": {},
    "agent4_output": {},
    "agent5_output": {},
}


def test_agent8_default_mode_is_automanifold():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm)
    agent.run(dict(_AGENT_CTX))
    prompt_used = llm.chat.call_args[1]["user_message"]
    assert "verified deterministic code" in prompt_used.lower()
    assert "hyperparameter selection mechanism" in prompt_used.lower()


def test_agent8_noskel_mode_uses_design_spec_not_primitive_agents():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="noskel_agents")
    agent.run({
        "design_spec": {
            "algorithm_name": "MyAlgo",
            "description": "Uses spectral methods",
            "mathematical_formulation": "L = D - W",
        },
        "agent1_output": {"data_summary": "high dimensional"},
    })
    prompt_used = llm.chat.call_args[1]["user_message"]
    assert "MyAlgo" in prompt_used
    assert "verified deterministic components" in prompt_used.lower()
    assert "dynamic components" in prompt_used.lower()


def test_agent8_noskel_returns_code():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="noskel_agents")
    result = agent.run({"design_spec": {}, "agent1_output": {}})
    assert "custom_embedding_code" in result
    assert "custom_embedding" in result["custom_embedding_code"]


@pytest.mark.parametrize("mode", ["novel", "all_llm", "novel_v2", "all_llm_v2", "bogus"])
def test_agent8_rejects_unknown_modes(mode):
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    with pytest.raises(ValueError):
        AlgorithmSynthesizerAgent(_make_llm(_SYNTH_RESPONSE), mode=mode)
