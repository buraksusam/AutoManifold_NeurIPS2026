"""ToolRegistry — name → schema + dispatch handler for the agent loop."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds tool schemas + dispatch handlers for the agent loop.

    Tools are registered with a name, an Anthropic-format schema dict
    (``description`` + ``input_schema``) and a handler callable
    ``handler(arguments: dict, session_state) -> dict``.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {}

    def register(
        self,
        name: str,
        schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any], Any], Dict[str, Any]],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        # We accept either {"description":..., "input_schema":...} or a full
        # {"name":..., "description":..., "input_schema":...}; strip "name".
        cleaned = {k: v for k, v in schema.items() if k != "name"}
        cleaned.setdefault("description", "")
        cleaned.setdefault("input_schema", {"type": "object", "properties": {}, "required": []})
        self._tools[name] = cleaned
        self._handlers[name] = handler

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas in our internal (Anthropic-native) format."""
        return [{"name": n, **s} for n, s in self._tools.items()]

    def dispatch(
        self,
        name: str,
        arguments: Dict[str, Any],
        session_state: Any,
    ) -> Dict[str, Any]:
        """Invoke the handler. Catch exceptions and surface as an error dict."""
        if name not in self._handlers:
            return {"error": f"unknown tool '{name}'", "tool_name": name}
        try:
            result = self._handlers[name](arguments or {}, session_state)
            if not isinstance(result, dict):
                # Wrap non-dict returns so the model always gets a dict
                return {"result": result}
            return result
        except Exception as exc:
            logger.warning("Tool '%s' raised: %s", name, exc)
            return {
                "error": str(exc),
                "tool_name": name,
                "traceback": traceback.format_exc(limit=3),
            }
