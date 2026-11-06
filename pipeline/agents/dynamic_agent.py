"""
DynamicAgent — a generic on-the-fly agent instantiated at runtime with a
custom role, system message, and task prompt.

Used by AllLLMDesignerAgent to spawn analysis sub-agents on demand without
any pre-defined agent library entry.
"""
from ..base_agent import BaseAgent
from ..llm_client import LLMClient


class DynamicAgent(BaseAgent):
    """Generic agent instantiated at runtime with a custom role and prompt."""

    AGENT_ID = 99  # sentinel — not a registered pipeline step

    def __init__(self, llm_client: LLMClient, role: str, system_msg: str, task_prompt: str):
        self.AGENT_NAME = f"Dynamic:{role}"   # set before super() reads it for logger
        super().__init__(llm_client)
        self._role = role
        self._system_msg = system_msg
        self._task_prompt = task_prompt

    def system_message(self) -> str:
        return self._system_msg

    def build_prompt(self, context: dict) -> str:
        # context is intentionally unused; the full prompt is supplied at instantiation
        return self._task_prompt
