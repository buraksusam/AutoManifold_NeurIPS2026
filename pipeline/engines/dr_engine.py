"""
DR & Visualization Engine

Computes 2D embeddings from high-dimensional data using a DR algorithm
with a given hyperparameter configuration, and produces scatter-plot
visualizations. Corresponds to the "DR & Visualization Engine" block in
the paper's Figure 1.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ..config import OUTPUT_ROOT

logger = logging.getLogger(__name__)


class DRVisualizationEngine:
    """Compute embeddings and render scatter plots."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_embedding(
        self,
        data: np.ndarray,
        method: str,
        params: Optional[dict] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Run a DR algorithm and return 2D coordinates.

        Args:
            data:   (n, D) high-dimensional array.
            method: One of 'tsne', 'umap', 'pacmap', 'pca'.
            params: Hyperparameter dict for the chosen method.
                    Any extra keyword arguments are merged into params.
            labels: Optional labels (unused by DR, kept for convenience).

        Returns:
            (n, 2) embedding array.
        """
        # Merge params dict and any extra kwargs so callers can use either style
        merged: dict = {}
        if params:
            merged.update(params)
        merged.update(kwargs)

        # PCA preprocessing (n_pcs)
        n_pcs = merged.get("n_pcs")
        if n_pcs and n_pcs < data.shape[1]:
            logger.info(f"PCA preprocessing: {data.shape[1]}D → {n_pcs}D")
            data = PCA(n_components=n_pcs).fit_transform(data)

        method = method.lower()
        if method == "tsne":
            return self._run_tsne(data, merged)
        elif method == "umap":
            return self._run_umap(data, merged)
        elif method == "pacmap":
            return self._run_pacmap(data, merged)
        elif method == "pca":
            return PCA(n_components=2).fit_transform(data)
        else:
            raise ValueError(f"Unsupported DR method: {method}")

    # High-contrast, colorblind-friendly palette (up to 12 classes)
    _PALETTE = [
        "#E74C3C", "#2E86AB", "#27AE60", "#F39C12", "#8E44AD",
        "#16A085", "#D35400", "#2C3E50", "#C0392B", "#1ABC9C",
        "#F1C40F", "#7F8C8D",
    ]

    def render_plot(
        self,
        embedding: np.ndarray,
        labels: Optional[np.ndarray] = None,
        label_names: Optional[dict] = None,
        title: str = "DR Embedding",
        quality_score: Optional[float] = None,
        save_path: Optional[str | Path] = None,
    ) -> str:
        """
        Produce a scatter plot of the 2D embedding.

        Returns:
            Path to the saved image file.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            for i, lab in enumerate(unique_labels):
                mask = labels == lab
                display_name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
                color = self._PALETTE[i % len(self._PALETTE)]
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=color,
                    label=f"{display_name} (n={mask.sum()})",
                    s=8,
                    alpha=0.7,
                    edgecolors="none",
                )
            ax.legend(
                fontsize=9,
                markerscale=2.5,
                loc="upper right",
                framealpha=0.85,
            )
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.7,
                       c="#2E86AB", edgecolors="none")

        # Quality score annotation
        if quality_score is not None:
            ax.text(
                0.02, 0.98,
                f"Quality Score: {quality_score:.2f} / 10",
                transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                          edgecolor="#F39C12", alpha=0.9),
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(OUTPUT_ROOT, "plots", "embedding.png")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved to {save_path}")
        return str(save_path)

    # ------------------------------------------------------------------
    # Private — DR algorithm runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_tsne(data: np.ndarray, params: dict) -> np.ndarray:
        from sklearn.manifold import TSNE
        import sklearn

        n_iter_val = params.get("max_iter", 1000)
        major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
        use_max_iter = (major > 1) or (major == 1 and minor >= 4)

        tsne_kwargs = dict(
            n_components=2,
            perplexity=float(params.get("perplexity", 30.0)),
            learning_rate=params.get("learning_rate", 200.0),
            early_exaggeration=float(params.get("early_exaggeration", 12.0)),
            angle=float(params.get("angle", 0.5)),
            init=params.get("init", "pca"),
            metric=params.get("metric", "euclidean"),
            n_iter_without_progress=int(params.get("n_iter_without_progress", 300)),
            random_state=42,
        )
        if use_max_iter:
            tsne_kwargs["max_iter"] = int(n_iter_val)
        else:
            tsne_kwargs["n_iter"] = int(n_iter_val)

        return TSNE(**tsne_kwargs).fit_transform(data)

    @staticmethod
    def _run_umap(data: np.ndarray, params: dict) -> np.ndarray:
        import umap

        # n_epochs=0 means "let UMAP choose automatically"
        n_epochs = params.get("n_epochs", 0) or None

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(params.get("n_neighbors", 15)),
            min_dist=float(params.get("min_dist", 0.1)),
            metric=params.get("metric", "euclidean"),
            n_epochs=n_epochs,
            learning_rate=float(params.get("learning_rate", 1.0)),
            spread=float(params.get("spread", 1.0)),
            negative_sample_rate=int(params.get("negative_sample_rate", 5)),
            repulsion_strength=float(params.get("repulsion_strength", 1.0)),
            set_op_mix_ratio=float(params.get("set_op_mix_ratio", 1.0)),
            local_connectivity=float(params.get("local_connectivity", 1.0)),
            random_state=42,
            n_jobs=1,  # explicit — avoids "n_jobs overridden to 1" warning when random_state is set
        )
        return reducer.fit_transform(data)

    @staticmethod
    def _run_pacmap(data: np.ndarray, params: dict) -> np.ndarray:
        import pacmap

        # num_iters can be stored as a tuple or as three separate keys
        if "num_iters" in params and isinstance(params["num_iters"], (list, tuple)):
            num_iters = tuple(int(x) for x in params["num_iters"])
        else:
            num_iters = (
                int(params.get("num_iters_phase1", 100)),
                int(params.get("num_iters_phase2", 100)),
                int(params.get("num_iters_phase3", 250)),
            )

        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=int(params.get("n_neighbors", 10)),
            MN_ratio=float(params.get("MN_ratio", 0.5)),
            FP_ratio=float(params.get("FP_ratio", 2.0)),
            lr=float(params.get("lr", 0.01)),
            num_iters=num_iters,
            apply_pca=bool(params.get("apply_pca", True)),
            distance=params.get("distance", "euclidean"),
            random_state=42,
        )
        return reducer.fit_transform(data)
