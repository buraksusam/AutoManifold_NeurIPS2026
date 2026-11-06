def test_all_llm_mode_imports_correct_planner():
    """When mode == 'all_llm', _run_pipeline must use AllLLMPlanner."""
    import app
    import inspect
    src = inspect.getsource(app._run_pipeline)
    assert "all_llm" in src
    assert "AllLLMPlanner" in src


def test_all_llm_planner_importable_from_pipeline():
    from pipeline.planner_all_llm import AllLLMPlanner
    assert AllLLMPlanner is not None


def test_v2_modes_are_routed_in_run_pipeline():
    import app
    import inspect
    src = inspect.getsource(app._run_pipeline)
    assert "novelv2" in src
    assert "allLLMv2" in src
    assert "NovelPlannerV2" in src
    assert "AllLLMPlannerV2" in src
