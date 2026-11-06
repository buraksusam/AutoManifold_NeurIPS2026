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
