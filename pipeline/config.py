"""
Pipeline configuration: LLM settings, default parameters, convergence criteria.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Where run artifacts and uploaded files are written. Defaults to
# <repo>/outputs and <repo>/uploads; override with environment variables.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = os.environ.get("AUTOMANIFOLD_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
UPLOAD_ROOT = os.environ.get("AUTOMANIFOLD_UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))


@dataclass
class LLMConfig:
    """Configuration for the LLM backend used by all agents."""

    provider: str = "openai"  # "openai", "anthropic", "google", "deepseek"
    model: str = "gpt-5.4"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None   # None = let the model use its native limit
    # Per-request HTTP timeout (seconds). Applied to every SDK call so the
    # AgentLoop can never hang silently when the provider stalls. 600s covers
    # opus + extended-thinking calls; bump higher for very long generations.
    timeout: int = 600
    # When set, activate the provider's reasoning/thinking mode.
    #   None     → no chain-of-thought streaming (legacy default).
    #   "low"    → small thinking budget.
    #   "medium" → moderate thinking budget.
    #   "high"   → maximum thinking budget.
    # DeepSeek thinking models accept this directly via `reasoning_effort`.
    # Anthropic thinking-capable models map it to a `thinking.budget_tokens` value.
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        if self.api_key is None:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }
            self.api_key = os.getenv(env_map.get(self.provider, "OPENAI_API_KEY"))


@dataclass
class ConvergenceConfig:
    """
    Convergence criteria derived from the paper's experimental results.
    Paper observed convergence in 3-9 iterations across models.
    """

    max_iterations: int = 10
    quality_score_target: float = 9.0
    min_improvement_threshold: float = 0.1  # stop if < 0.1 improvement over patience window
    patience: int = 2  # number of iterations to wait for improvement
    trustworthiness_target: float = 0.95
    continuity_target: float = 0.95
    # When False, skip Agent 6/7 hyperparameter-optimization loop entirely
    # (novel/all_llm modes do a single-shot embedding; optimization mode is capped to 1 iter).
    enable_iterative_optimization: bool = True


@dataclass
class PatchConfig:
    """Default parameters for patching in Sections 1.1 and 1.2 of DR_Template."""

    target_patch_size: int = 100  # m ≈ 100 points per patch (DR_Template §1.1)
    reliability_threshold_T: float = 0.90
    reliability_threshold_C: float = 0.90
    reliability_threshold_S: float = 0.30  # Stress upper bound
    neighborhood_k_values: list = field(
        default_factory=lambda: [10, 20, 30, 50, 75, 100, 150, 200]
    )


@dataclass
class MetricsConfig:
    """Default neighbourhood sizes for metric computation."""

    neighborhood_k: int = 10  # k for T(k) and C(k)
    lof_neighbors: int = 20  # k for LOF
    lof_outlier_threshold: float = 1.5
    silhouette_sample_size: int = 5000  # subsample for speed


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Default DR algorithm to start with
    default_dr_method: str = "umap"
    default_hyperparameters: dict = field(
        default_factory=lambda: {
            "tsne": {
                "perplexity": 30.0,
                "learning_rate": 200.0,
                "max_iter": 1000,
                "early_exaggeration": 12.0,
                "angle": 0.5,
                "init": "pca",
                "metric": "euclidean",
                "n_iter_without_progress": 300,
                "n_pcs": 20,
            },
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "n_epochs": None,
                "learning_rate": 1.0,
                "spread": 1.0,
                "negative_sample_rate": 5,
                "repulsion_strength": 1.0,
                "set_op_mix_ratio": 1.0,
                "local_connectivity": 1.0,
                "n_pcs": 20,
            },
            "pacmap": {
                "n_neighbors": 10,
                "MN_ratio": 0.5,
                "FP_ratio": 2.0,
                "lr": 0.01,
                "num_iters": (100, 100, 250),
                "apply_pca": True,
                "distance": "euclidean",
                "n_pcs": 20,
            },
        }
    )

    # Design goal passed to agents
    primary_goal: str = "balance both"  # "local structure" | "global structure" | "balance both"

    # Top-level hyperparameters (override default_hyperparameters per-method)
    n_pcs: int = 20

    # t-SNE
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_max_iter: int = 1000
    tsne_early_exaggeration: float = 12.0
    tsne_angle: float = 0.5
    tsne_init: str = "pca"               # 'pca' | 'random'
    tsne_metric: str = "euclidean"
    tsne_n_iter_without_progress: int = 300

    # UMAP
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_n_epochs: int = 0               # 0 = auto (UMAP default)
    umap_learning_rate: float = 1.0
    umap_spread: float = 1.0
    umap_negative_sample_rate: int = 5
    umap_repulsion_strength: float = 1.0
    umap_set_op_mix_ratio: float = 1.0
    umap_local_connectivity: float = 1.0

    # PaCMAP
    pacmap_n_neighbors: int = 10
    pacmap_MN_ratio: float = 0.5
    pacmap_FP_ratio: float = 2.0
    pacmap_lr: float = 0.01
    pacmap_num_iters_phase1: int = 100   # phase-1 iterations (attraction)
    pacmap_num_iters_phase2: int = 100   # phase-2 iterations (mixed)
    pacmap_num_iters_phase3: int = 250   # phase-3 iterations (repulsion)
    pacmap_apply_pca: bool = True
    pacmap_distance: str = "euclidean"

    # Output directories
    output_dir: str = OUTPUT_ROOT
    plots_dir: str = os.path.join(OUTPUT_ROOT, "plots")
    logs_dir: str = os.path.join(OUTPUT_ROOT, "logs")

    # All LLM mode — Agent 10 retry budget (higher than novel mode's default of 2)
    all_llm_max_recovery_attempts: int = 5
