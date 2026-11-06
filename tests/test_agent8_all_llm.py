import json
from unittest.mock import MagicMock


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


def test_agent8_all_llm_mode_uses_design_spec_not_primitive_agents():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="all_llm")
    ctx = {
        "mode": "all_llm",
        "design_spec": {
            "algorithm_name": "MyAlgo",
            "description": "Uses spectral methods",
            "mathematical_formulation": "L = D - W",
        },
        "agent1_output": {"data_summary": "high dimensional"},
    }
    result = agent.run(ctx)
    prompt_used = llm.chat.call_args[1]["user_message"]
    # all_llm prompt should reference design_spec, not primitive agents
    assert "design_spec" in prompt_used.lower() or "algorithm_name" in prompt_used.lower() or "MyAlgo" in prompt_used


def test_agent8_default_mode_unchanged():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm)  # default mode = "novel"
    ctx = {
        "agent1_output": {"summary": "data"},
        "agent2_output": {},
        "agent3_output": {},
        "agent4_output": {},
        "agent5_output": {},
    }
    result = agent.run(ctx)
    prompt_used = llm.chat.call_args[1]["user_message"]
    # Original prompt references primitive strategies
    assert "Data Feeding" in prompt_used or "Strategy" in prompt_used


def test_agent8_all_llm_returns_code():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="all_llm")
    result = agent.run({"mode": "all_llm", "design_spec": {}, "agent1_output": {}})
    assert "custom_embedding_code" in result
    assert "custom_embedding" in result["custom_embedding_code"]


def test_agent8_novel_v2_prompt_mentions_verified_code_constraints():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="novel_v2")
    agent.run({
        "agent1_output": {"summary": "data"},
        "agent2_output": {},
        "agent3_output": {},
        "agent4_output": {},
        "agent5_output": {},
    })
    prompt_used = llm.chat.call_args[1]["user_message"]
    assert "verified deterministic code" in prompt_used.lower()
    assert "hyperparameter selection mechanism" in prompt_used.lower()


def test_agent8_all_llm_v2_prompt_mentions_code_classification_constraints():
    from pipeline.agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
    llm = _make_llm(_SYNTH_RESPONSE)
    agent = AlgorithmSynthesizerAgent(llm, mode="all_llm_v2")
    agent.run({"mode": "allLLMv2", "design_spec": {}, "agent1_output": {}})
    prompt_used = llm.chat.call_args[1]["user_message"]
    assert "verified deterministic components" in prompt_used.lower()
    assert "dynamic components" in prompt_used.lower()
