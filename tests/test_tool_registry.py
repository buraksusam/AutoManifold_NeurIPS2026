"""Tests for ToolRegistry."""

from pipeline.tools.registry import ToolRegistry


def test_registry_register_and_dispatch():
    reg = ToolRegistry()
    captured = {}

    def handler(args, state):
        captured["args"] = args
        captured["state"] = state
        return {"ok": True, "echo": args.get("x")}

    reg.register(
        "do_thing",
        {
            "description": "test tool",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": []},
        },
        handler,
    )
    out = reg.dispatch("do_thing", {"x": "hello"}, session_state="STATE")
    assert out == {"ok": True, "echo": "hello"}
    assert captured["args"] == {"x": "hello"}
    assert captured["state"] == "STATE"


def test_registry_dispatch_unknown_tool_returns_error():
    reg = ToolRegistry()
    out = reg.dispatch("nope", {}, None)
    assert "error" in out and "unknown" in out["error"].lower()
    assert out["tool_name"] == "nope"


def test_registry_handler_exception_caught():
    reg = ToolRegistry()

    def handler(args, state):
        raise RuntimeError("boom")

    reg.register("bad", {"description": "x"}, handler)
    out = reg.dispatch("bad", {}, None)
    assert "error" in out
    assert "boom" in out["error"]
    assert out["tool_name"] == "bad"


def test_registry_schemas_format():
    reg = ToolRegistry()
    reg.register(
        "t1",
        {"description": "d1", "input_schema": {"type": "object", "properties": {}, "required": []}},
        lambda args, st: {},
    )
    reg.register(
        "t2",
        {"description": "d2", "input_schema": {"type": "object", "properties": {}, "required": []}},
        lambda args, st: {},
    )
    schemas = reg.schemas()
    names = {s["name"] for s in schemas}
    assert names == {"t1", "t2"}
    for s in schemas:
        assert "description" in s and "input_schema" in s


def test_registry_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register("a", {"description": "x"}, lambda args, st: {})
    try:
        reg.register("a", {"description": "y"}, lambda args, st: {})
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError")
