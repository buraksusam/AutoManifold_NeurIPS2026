"""Skill tools — wrap each `SKILL.md` as an agent-loop tool.

Each tool's name + description come from the SKILL.md frontmatter; the body
(post-frontmatter) is returned as the tool result so the LLM can produce the
required structured output in its next turn.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_skill_md(path: Path) -> Tuple[str, str, str]:
    """Parse a SKILL.md file into (name, description, body).

    Frontmatter format::

        ---
        name: skill-name
        description: |
          A multi-line description.
        ---
        <body>
    """
    text = path.read_text()
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"SKILL.md missing frontmatter: {path}")
    fm_text, body = m.group(1), m.group(2).strip()

    name: str = ""
    description_lines: List[str] = []
    in_description = False
    desc_indent: int = -1

    for line in fm_text.splitlines():
        if not line.strip():
            if in_description:
                description_lines.append("")
            continue
        # Detect a new top-level key
        stripped = line.lstrip()
        leading = len(line) - len(stripped)
        if not in_description and ":" in stripped and leading == 0:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "name":
                # Strip optional quotes
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                name = val
                in_description = False
            elif key == "description":
                in_description = True
                desc_indent = -1
                if val and val not in ("|", ">"):
                    description_lines.append(val)
            else:
                in_description = False
            continue
        if in_description:
            # First continuation line establishes the indent
            if desc_indent < 0:
                desc_indent = leading
            cleaned = line[desc_indent:] if leading >= desc_indent else stripped
            description_lines.append(cleaned)

    description = "\n".join(description_lines).strip()
    if not name:
        raise ValueError(f"SKILL.md missing 'name' field: {path}")
    return name, description, body


def load_skills_from_dir(path: str | Path) -> List[Tuple[str, str, str]]:
    """Walk ``<path>/<skill>/SKILL.md`` and return list of (name, description, body)."""
    base = Path(path)
    if not base.is_dir():
        raise ValueError(f"skills dir not found: {base}")
    out: List[Tuple[str, str, str]] = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        skill_file = sub / "SKILL.md"
        if not skill_file.exists():
            continue
        name, desc, body = _parse_skill_md(skill_file)
        out.append((name, desc, body))
    return out


def _extract_required_output_schema(body: str) -> str:
    """Pull the 'REQUIRED OUTPUT (JSON)' block out of a SKILL.md body.

    Returns an empty string if no such block is present.
    """
    if not body:
        return ""
    # Look for the marker "REQUIRED OUTPUT" (case-insensitive) and grab the
    # rest of the body (usually ends at end-of-file).
    m = re.search(r"REQUIRED\s+OUTPUT[^\n]*\n", body, re.IGNORECASE)
    if not m:
        return ""
    return body[m.start():].strip()


def _make_skill_handler(skill_name: str, body: str):
    expected_schema = _extract_required_output_schema(body)

    def _handler(args: Dict[str, Any], state) -> Dict[str, Any]:
        task_context = args.get("task_context", "") or ""
        return {
            "skill_name": skill_name,
            "task_context_echo": task_context,
            "skill_body": body,
            "expected_output_schema": expected_schema,
        }
    return _handler


def register_skill_tools(registry: ToolRegistry, skills_dir: str | Path) -> List[str]:
    """Discover and register every SKILL.md under skills_dir as a tool.

    Returns the list of registered skill names.
    """
    registered: List[str] = []
    for name, description, body in load_skills_from_dir(skills_dir):
        # Truncate very long descriptions for the API (Anthropic caps tool descriptions).
        desc = description if description else f"Skill: {name}"
        schema = {
            "description": desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_context": {
                        "type": "string",
                        "description": (
                            "Brief description of the scope/inputs you want this skill "
                            "to address (e.g., which prior outputs to focus on)."
                        ),
                    },
                },
                "required": ["task_context"],
            },
        }
        registry.register(name, schema, _make_skill_handler(name, body))
        registered.append(name)
    return registered
