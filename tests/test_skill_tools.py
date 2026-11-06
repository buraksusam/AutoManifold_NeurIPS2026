"""Tests for skill_tools — discovery + dispatch."""

import os

from pipeline.tools import ToolRegistry, register_skill_tools
from pipeline.tools.skill_tools import load_skills_from_dir

SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pipeline", "skills", "auto_skills",
)


EXPECTED_SKILLS = {
    "analyze-data",
    "map-difficulty-to-primitives",
    "estimate-neighborhoods",
    "design-affinity",
    "compose-cost",
    "validate-novelty",
    "synthesize-algorithm",
    "recover-failing-code",
    "describe-math",
    "interpret-forensics",
}


def test_skill_tools_load_all_10_skills_from_dir():
    skills = load_skills_from_dir(SKILLS_DIR)
    names = {s[0] for s in skills}
    assert names == EXPECTED_SKILLS, f"got {names}"
    for _, desc, body in skills:
        assert desc.strip(), "description must be non-empty"
        assert body.strip(), "body must be non-empty"


def test_skill_tool_dispatch_returns_body():
    reg = ToolRegistry()
    registered = register_skill_tools(reg, SKILLS_DIR)
    assert set(registered) == EXPECTED_SKILLS
    out = reg.dispatch(
        "analyze-data",
        {"task_context": "characterize my dataset's geometry"},
        session_state=None,
    )
    assert out["task_context_echo"] == "characterize my dataset's geometry"
    assert out["skill_body"]
    # The body should contain the REQUIRED OUTPUT marker for analyze-data
    assert "REQUIRED OUTPUT" in out["skill_body"] or "REQUIRED" in out["skill_body"].upper()
    # expected_output_schema should be a non-empty extract
    assert out["expected_output_schema"]


def test_skill_tool_schemas_have_task_context_required():
    reg = ToolRegistry()
    register_skill_tools(reg, SKILLS_DIR)
    for schema in reg.schemas():
        if schema["name"] in EXPECTED_SKILLS:
            assert schema["input_schema"]["required"] == ["task_context"]
