"""
Dimensionality Reduction Algorithm Design Pipeline

An agentic framework that orchestrates specialized worker agents to design,
implement, evaluate, and optimize dimensionality reduction algorithms.

Based on:
  - DR_Template: a structured template for DR algorithm design
    (auxiliary tasks + implementation strategies) encoded in the agent prompts.
  - "Explainable Iterative Data Visualisation Refinement via an LLM Agent"
    (Susam & Mu, IEEE CG&A 2026)

Two Planning Modes:
  - Planner: Optimizes existing DR methods (t-SNE, UMAP, PaCMAP) through
    iterative LLM-guided refinement.
  - NovelPlanner: Generates novel DR algorithms through template-based
    composition of algorithmic primitives (affinity, cost, optimization).

Features:
  - Skill Tracking: Logs per-agent skill invocations for observability
"""

__version__ = "0.3.0"

from .planner import Planner
from .planner_novel import NovelPlanner
from .planner_novel_v2 import NovelPlannerV2
from .planner_all_llm import AllLLMPlanner
from .planner_all_llm_v2 import AllLLMPlannerV2

__all__ = [
    "Planner",
    "NovelPlanner",
  "NovelPlannerV2",
  "AllLLMPlanner",
  "AllLLMPlannerV2",
]

# Skill tracking exports
try:
    from .skill_tracker import (
        SkillTracker,
        track_skill,
        get_tracker,
        get_skill_summary,
    )
    __all__.extend([
        "SkillTracker",
        "track_skill",
        "get_tracker",
        "get_skill_summary",
    ])
except ImportError:
    pass  # Skill tracking optional
