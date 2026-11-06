import importlib


EXPECTED_MODES = {
    "optimization": "Planner",
    "automanifold": "AutoManifoldPlanner",
    "noskel_agents": "NoSkelAgentsPlanner",
    "auto_skills": "AutoSkillsPlanner",
    "singlellm": "SingleLLMPlanner",
}


def test_app_exposes_exactly_the_supported_modes():
    import app
    assert set(app.MODES) == set(EXPECTED_MODES)


def test_every_mode_resolves_to_its_planner_class():
    import app
    for mode, class_name in EXPECTED_MODES.items():
        module_name, cls, _display = app.MODES[mode]
        assert cls == class_name
        assert getattr(importlib.import_module(module_name), cls).__name__ == class_name


def test_planners_importable_from_package():
    from pipeline import (AutoManifoldPlanner, AutoSkillsPlanner, NoSkelAgentsPlanner,
                          Planner, SingleLLMPlanner)
    assert issubclass(NoSkelAgentsPlanner, AutoManifoldPlanner)
    assert issubclass(SingleLLMPlanner, AutoManifoldPlanner)
    assert Planner and AutoSkillsPlanner


def test_frontend_offers_exactly_the_supported_modes():
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text()
    select = html[html.index('<select id="mode-select"'):]
    select = select[:select.index("</select>")]
    assert set(re.findall(r'<option value="([^"]+)"', select)) == set(EXPECTED_MODES)


def test_run_pipeline_builds_config_and_runs_every_mode(monkeypatch, tmp_path):
    """Drive app._run_pipeline end-to-end for each mode with a stub planner."""
    import queue
    import sys
    import types

    import numpy as np

    import app

    seen = {}

    class StubPlanner:
        def __init__(self, config):
            seen["config"] = config

        def run(self, data, labels=None, label_names=None, output_dir=None):
            return {"ok": True}

    stub_module = types.ModuleType("stub_planner_module")
    stub_module.StubPlanner = StubPlanner
    monkeypatch.setitem(sys.modules, "stub_planner_module", stub_module)

    cfg = {"provider": "openai", "model": "m", "api_key": "k", "mode": None,
           "dr_method": "umap", "max_iter": 1, "quality_target": 9.0,
           "primary_goal": "balance both", "iterative_optimization": False}
    for mode in EXPECTED_MODES:
        monkeypatch.setitem(app.MODES, mode, ("stub_planner_module", "StubPlanner", mode))
        run_id = f"test_{mode}"
        app.runs[run_id] = {"q": queue.Queue(), "status": "starting"}
        app._run_pipeline(run_id, {**cfg, "mode": mode}, np.zeros((5, 2)), None, None, str(tmp_path))
        assert app.runs[run_id]["status"] == "complete", app.runs[run_id].get("error")
        assert seen["config"].llm.model == "m"
        assert seen["config"].default_dr_method == "umap"
        app.runs.pop(run_id)
