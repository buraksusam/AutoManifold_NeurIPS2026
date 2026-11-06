"""
Dimensionality Reduction Algorithm Design Pipeline

An agentic framework that orchestrates specialized worker agents to design,
implement, evaluate, and optimize dimensionality reduction algorithms.

Based on:
  - DR_Template (docs/DR_Template.md) and DR_Template_v2 (docs/DR_Template_v2.md):
    structured templates for DR algorithm design (auxiliary tasks,
    implementation strategies, synthesis constraints) encoded in the agent prompts.
  - "Explainable Iterative Data Visualisation Refinement via an LLM Agent"
    (Susam & Mu, IEEE CG&A 2026)

Pipeline modes:
  - Optimization (Planner): tunes existing DR methods (t-SNE, UMAP, PaCMAP)
    through iterative LLM-guided refinement.
  - AutoManifold (AutoManifoldPlanner): designs novel DR algorithms by
    template-based composition of algorithmic primitives.
  - NoSkel-Agents (NoSkelAgentsPlanner): skeleton-free, multi-agent design.
  - Auto-Skills (AutoSkillsPlanner): autonomous agent loop over skills.
  - SingleLLM (SingleLLMPlanner): single blank-prompt LLM baseline.

Features:
  - Skill Tracking: Logs per-agent skill invocations for observability
"""

__version__ = "0.3.0"

from .planner import Planner
from .planner_automanifold import AutoManifoldPlanner
from .planner_noskel_agents import NoSkelAgentsPlanner
from .planner_auto_skills import AutoSkillsPlanner
from .planner_singlellm import SingleLLMPlanner

__all__ = [
    "Planner",
    "AutoManifoldPlanner",
    "NoSkelAgentsPlanner",
    "AutoSkillsPlanner",
    "SingleLLMPlanner",
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
