"""Worker agents package — one module per specialized agent.

Exports the AutoManifold agent set. Optimization-mode agents
(PatchStitching, OutlierIdentification, DefaultComparison) are imported
directly by ``pipeline/planner.py`` and intentionally NOT re-exported here.
"""

from .agent_1_data_characterization import DataCharacterizationAgent
from .agent_2_difficulty_analyzer import EmbeddingDifficultyAgent
from .agent_3_neighborhood_analyzer import NeighborhoodAnalyzerAgent
from .agent_4_affinity_designer import AffinityDesignerAgent
from .agent_5_cost_composer import CostComposerAgent
from .agent_6_hyperparameter_opt import HyperparameterOptAgent
from .agent_12_novelty_validator import NoveltyValidatorAgent
from .agent_7_visualization_evaluator import VisualizationEvaluatorAgent
from .agent_8_synthesizer import AlgorithmSynthesizerAgent
from .agent_9_visual_forensics import VisualForensicsAgent
from .agent_10_error_recovery import ErrorRecoveryAgent
from .agent_11_math_pseudocode import MathPseudocodeAgent

__all__ = [
    "DataCharacterizationAgent",
    "EmbeddingDifficultyAgent",
    "NeighborhoodAnalyzerAgent",
    "AffinityDesignerAgent",
    "CostComposerAgent",
    "HyperparameterOptAgent",
    "NoveltyValidatorAgent",
    "VisualizationEvaluatorAgent",
    "AlgorithmSynthesizerAgent",
    "VisualForensicsAgent",
    "ErrorRecoveryAgent",
    "MathPseudocodeAgent",
]
