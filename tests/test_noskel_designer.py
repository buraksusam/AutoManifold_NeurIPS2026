import json
from unittest.mock import MagicMock, call


def _make_llm(*responses):
    mock = MagicMock()
    if len(responses) == 1:
        mock.chat.return_value = responses[0]
    else:
        mock.chat.side_effect = list(responses)
    return mock


def _finalize_response(name="TestAlgo"):
    return json.dumps({
        "action": "finalize_design",
        "algorithm_name": name,
        "description": "A test DR algorithm",
        "mathematical_formulation": "f(x) = Ax",
        "key_innovations": ["novel graph construction"],
        "suggested_params": {"lr": 0.01, "n_neighbors": 15},
        "design_rationale": "Suited to high-dimensional sparse data",
    })


def _spawn_response(role="Analyzer"):
    return json.dumps({
        "action": "spawn_agents",
        "reasoning": "need manifold analysis",
        "agents": [{
            "role": role,
            "system_message": f"You are a {role}.",
            "task_prompt": "Analyze the manifold structure and return JSON.",
        }],
    })


def test_designer_finalizes_on_first_round():
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    llm = _make_llm(_finalize_response("DirectAlgo"))
    agent = NoSkelDesignerAgent(llm)
    result = agent.run_design_loop({"data_char": "test"})
    assert result["action"] == "finalize_design"
    assert result["algorithm_name"] == "DirectAlgo"
    assert llm.chat.call_count == 1


def test_designer_spawns_sub_agent_then_finalizes():
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    # Round 1: spawn, sub-agent, Round 2: finalize
    llm = _make_llm(
        _spawn_response("ManifoldAnalyzer"),   # round 1 designer
        '{"manifold_type": "swiss_roll"}',     # DynamicAgent call
        _finalize_response("SpawnedAlgo"),     # round 2 designer
    )
    agent = NoSkelDesignerAgent(llm)
    result = agent.run_design_loop({})
    assert result["algorithm_name"] == "SpawnedAlgo"
    assert llm.chat.call_count == 3


def test_designer_forces_finalize_on_last_round():
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    spawn = _spawn_response()
    llm = _make_llm(
        spawn,                          # round 1 designer → spawn_agents
        '{"result": 1}',               # sub-agent round 1
        spawn,                          # round 2 designer → spawn_agents
        '{"result": 2}',               # sub-agent round 2
        spawn,                          # round 3 designer → spawn_agents (is_last=True, triggers _force_finalize)
        _finalize_response("Forced"),   # _force_finalize LLM call
    )
    agent = NoSkelDesignerAgent(llm)
    result = agent.run_design_loop({})
    assert "algorithm_name" in result
    assert result.get("_forced_finalize") is True
    assert len(result.get("_session_transcript", [])) == 3


def test_designer_handles_sub_agent_failure_gracefully():
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    llm = MagicMock()
    finalize = _finalize_response("ResilientAlgo")
    # Round 1: spawn_agents. Sub-agent call raises exception. Round 2: finalize.
    llm.chat.side_effect = [
        _spawn_response(),         # round 1 designer
        Exception("LLM timeout"),  # sub-agent call raises
        finalize,                  # round 2 designer
    ]
    agent = NoSkelDesignerAgent(llm)
    result = agent.run_design_loop({})
    assert result["algorithm_name"] == "ResilientAlgo"


def test_designer_embeds_sub_agent_results_in_next_round():
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    sub_result = '{"manifold_dim": 3, "structure": "curved"}'
    llm = _make_llm(
        _spawn_response("ManifoldDetector"),
        sub_result,
        _finalize_response(),
    )
    agent = NoSkelDesignerAgent(llm)
    agent.run_design_loop({})
    # Third call (round 2 designer) should have sub-agent results in prompt
    round2_prompt = llm.chat.call_args_list[2][1]["user_message"]
    assert "ManifoldDetector" in round2_prompt
    assert "manifold_dim" in round2_prompt
