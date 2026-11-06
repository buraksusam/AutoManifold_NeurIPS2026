from unittest.mock import MagicMock, patch
from pipeline.config import PipelineConfig, LLMConfig
import numpy as np


def _make_config():
    return PipelineConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="test-key"),
        noskel_max_recovery_attempts=5,
    )


def test_noskel_planner_can_be_instantiated():
    from pipeline.planner_noskel_agents import NoSkelAgentsPlanner
    planner = NoSkelAgentsPlanner(_make_config())
    assert planner is not None


def test_noskel_planner_has_designer_agent():
    from pipeline.planner_noskel_agents import NoSkelAgentsPlanner
    from pipeline.agents.agent_noskel_designer import NoSkelDesignerAgent
    planner = NoSkelAgentsPlanner(_make_config())
    assert hasattr(planner, "designer_agent")
    assert isinstance(planner.designer_agent, NoSkelDesignerAgent)


def test_noskel_planner_has_agent10():
    from pipeline.planner_noskel_agents import NoSkelAgentsPlanner
    from pipeline.agents.agent_10_error_recovery import ErrorRecoveryAgent
    planner = NoSkelAgentsPlanner(_make_config())
    assert hasattr(planner, "agent10")
    assert isinstance(planner.agent10, ErrorRecoveryAgent)


def test_noskel_planner_phase4_uses_higher_retry_budget(tmp_path):
    """NoSkelAgentsPlanner._phase4_implement_and_recover must pass max_attempts=5 to _attempt_error_recovery."""
    from pipeline.planner_noskel_agents import NoSkelAgentsPlanner

    planner = NoSkelAgentsPlanner(_make_config())
    planner.output_dir = str(tmp_path)

    # Code that always fails
    bad_code = "def custom_embedding(data, labels, params):\n    raise RuntimeError('always fails')"
    synth_result = {
        "custom_embedding_code": bad_code,
        "default_params": {},
        "algorithm_specification": {"name": "TestAlgo"},
    }

    # Mock _attempt_error_recovery to capture max_attempts
    captured = {}
    def mock_recovery(*args, **kwargs):
        captured["max_attempts"] = kwargs.get("max_attempts", args[5] if len(args) > 5 else None)
        return bad_code, {}, False  # always fail recovery
    planner._attempt_error_recovery = mock_recovery

    # Mock dr_engine fallback
    fake_emb = np.zeros((10, 2))
    planner.dr_engine.compute_embedding = MagicMock(return_value=fake_emb)
    planner.context = {"design_spec": {}}

    data = np.random.rand(10, 5)
    planner._phase4_implement_and_recover(data, None, None, synth_result)
    assert captured.get("max_attempts") == 5


def test_noskel_run_produces_baseline_comparison(tmp_path):
    from pipeline.planner_noskel_agents import NoSkelAgentsPlanner

    planner = NoSkelAgentsPlanner(_make_config())
    data = np.random.rand(12, 4)
    embedding = np.random.rand(12, 2)

    planner._phase1_characterize = MagicMock()
    planner._phase2_design = MagicMock(return_value={"algorithm_name": "BlankCanvas"})
    planner._phase3_synthesize_noskel = MagicMock(return_value={"algorithm_specification": {"name": "BlankCanvas"}})
    planner._phase4_implement_and_recover = MagicMock(return_value={
        "algo_name": "BlankCanvas",
        "used_fallback": False,
        "code": "def custom_embedding(data, labels, params):\n    return data[:, :2]",
        "params": {},
    })

    def fake_iterative(*args, **kwargs):
        planner.best_result = {
            "iteration": 1,
            "embedding": embedding,
            "metrics": {"trustworthiness": 0.9},
            "hyperparameters": {},
            "dendrograms": {},
        }
        return embedding

    planner._iterative_optimise = MagicMock(side_effect=fake_iterative)
    planner._compare_with_baselines = MagicMock(return_value={"comparison_table": [{"method": "★ BlankCanvas"}]})
    planner._phase7_visual_forensics = MagicMock(return_value={"status": "ok"})

    result = planner.run(data, output_dir=str(tmp_path))

    planner._compare_with_baselines.assert_called_once()
    assert result["baseline_comparison"]["comparison_table"][0]["method"] == "★ BlankCanvas"
    assert (tmp_path / "final_noskel_report.json").exists()
