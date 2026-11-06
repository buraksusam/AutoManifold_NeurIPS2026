"""AllLLMPlannerV2 — template-v2 variant of the blank-canvas design pipeline."""

from .planner_all_llm import AllLLMPlanner


class AllLLMPlannerV2(AllLLMPlanner):
    """All-LLM planner variant with DR_Template_v2-aware design and synthesis prompts."""

    def __init__(self, config):
        super().__init__(config)
        from .agents.agent_8_synthesizer import AlgorithmSynthesizerAgent
        from .agents.agent_all_llm_designer_v2 import AllLLMDesignerAgentV2

        self.mode_name = "allLLMv2"
        self.template_name = "DR_Template_v2"
        self.synth_mode = "all_llm_v2"
        self.designer_agent = AllLLMDesignerAgentV2(self.llm)
        self.agent8 = AlgorithmSynthesizerAgent(self.llm, mode="all_llm_v2")