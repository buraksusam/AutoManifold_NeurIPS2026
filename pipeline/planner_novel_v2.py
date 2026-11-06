"""NovelPlannerV2 — template-v2 variant of the novel algorithm pipeline."""

from .planner_novel import NovelPlanner


class NovelPlannerV2(NovelPlanner):
    """Novel planner variant that uses DR_Template_v2 synthesis constraints."""

    def __init__(self, config):
        super().__init__(config)
        from .agents.agent_8_synthesizer import AlgorithmSynthesizerAgent

        self.mode_name = "novelv2"
        self.template_name = "DR_Template_v2"
        self.agent8 = AlgorithmSynthesizerAgent(self.llm, mode="novel_v2")

    # Phase 6 full-data run inherits NovelPlanner._run_custom_code, which now
    # has no timeout — long-running novel algorithms run to completion instead
    # of being interrupted and restarted.
