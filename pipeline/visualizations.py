"""
Auxiliary visualisation utilities.

Produces diagnostic plots at each pipeline phase and saves them to the
experiment output directory.  Keeps matplotlib usage out of the planner.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from .config import OUTPUT_ROOT


# ── Shared palette ──────────────────────────────────────────────────────────
_PALETTE = [
    "#E74C3C", "#2E86AB", "#27AE60", "#F39C12", "#8E44AD",
    "#16A085", "#D35400", "#2C3E50", "#C0392B", "#1ABC9C",
    "#F1C40F", "#7F8C8D",
]

# ── Phase 1 ─────────────────────────────────────────────────────────────────

def plot_data_overview(
    data: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    pca_model,          # fitted sklearn PCA object
    output_dir: str,
) -> str:
    """
    2×2 figure:
      [0,0] PCA explained variance curve
      [0,1] PCA 2D scatter (with labels if available)
      [1,0] Per-feature std-dev bar
      [1,1] Label distribution bar (or blank if no labels)
    """
    from sklearn.decomposition import PCA as _PCA

    n_components = len(pca_model.explained_variance_ratio_)
    cumvar = np.cumsum(pca_model.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Data Overview  —  {data.shape[0]} points × {data.shape[1]} features",
                 fontsize=14, fontweight="bold")

    # ── [0,0] Explained variance ──
    ax = axes[0, 0]
    xs = np.arange(1, n_components + 1)
    ax.plot(xs, cumvar, color="#2E86AB", linewidth=2)
    ax.fill_between(xs, cumvar, alpha=0.15, color="#2E86AB")
    for thresh, col in [(80, "#27AE60"), (90, "#F39C12"), (95, "#E74C3C")]:
        idx = np.searchsorted(cumvar, thresh)
        if idx < len(cumvar):
            ax.axhline(thresh, linestyle="--", color=col, linewidth=0.8, alpha=0.7)
            ax.annotate(f"{thresh}% at PC{idx+1}", xy=(idx + 1, thresh),
                        fontsize=7, color=col, va="bottom")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Explained Variance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(1, min(n_components, 50))

    # ── [0,1] PCA 2D scatter ──
    ax = axes[0, 1]
    pca2 = _PCA(n_components=2).fit_transform(data)
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
            ax.scatter(pca2[mask, 0], pca2[mask, 1],
                       c=_PALETTE[i % len(_PALETTE)],
                       label=f"{name} ({mask.sum()})", s=6, alpha=0.65, edgecolors="none")
        ax.legend(fontsize=8, markerscale=2.5, framealpha=0.8)
    else:
        ax.scatter(pca2[:, 0], pca2[:, 1], s=6, alpha=0.6, c="#2E86AB", edgecolors="none")
    ax.set_title("PCA 2D Projection")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── [1,0] Feature std-devs (top-30 by std) ──
    ax = axes[1, 0]
    stds = np.std(data, axis=0)
    top_idx = np.argsort(stds)[::-1][:30]
    ax.bar(np.arange(len(top_idx)), stds[top_idx], color="#8E44AD", alpha=0.75)
    ax.set_xlabel("Feature rank (by std)")
    ax.set_ylabel("Std Dev")
    ax.set_title("Top-30 Feature Std Devs")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # ── [1,1] Label distribution ──
    ax = axes[1, 1]
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        names = [label_names.get(int(l), str(l)) if label_names else str(l)
                 for l in unique_labels]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(unique_labels))]
        bars = ax.bar(names, counts, color=colors, alpha=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{cnt}\n({100*cnt/len(labels):.1f}%)", ha="center", fontsize=8)
        ax.set_title("Class Distribution")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No labels provided", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Class Distribution")

    fig.tight_layout()
    path = os.path.join(output_dir, "phase1_data_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Phase 2a — Difficulty map ────────────────────────────────────────────────

def plot_difficulty_map(
    embedding: np.ndarray,          # pre-computed 2D embedding (n, 2)
    point_difficulty: np.ndarray,   # per-point label: "H", "M", "E"
    patch_reliability: np.ndarray,  # per-point composite reliability score 0-1
    output_dir: str,
    method: str = "DR",             # name shown in titles / axis labels
) -> str:
    """
    Two-panel figure on the preferred-method 2D embedding:
      Left:  points coloured by H / M / E difficulty category
      Right: points coloured by continuous reliability score (heat-map)
    """
    method_label = method.upper()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Embedding Difficulty Map ({method_label} projection)",
                 fontsize=13, fontweight="bold")

    # ── Left: categorical H/M/E ──
    ax = axes[0]
    diff_map = {"H": ("#E74C3C", "Hard"), "M": ("#F39C12", "Moderate"), "E": ("#27AE60", "Easy")}
    for cat, (col, label) in diff_map.items():
        mask = point_difficulty == cat
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=col,
                       label=f"{label} ({mask.sum()})",
                       s=8, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=9, markerscale=2.5, framealpha=0.85)
    ax.set_title("H / M / E Difficulty Categories")
    ax.set_xlabel(f"{method_label} Dim 1"); ax.set_ylabel(f"{method_label} Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Right: continuous reliability score ──
    ax = axes[1]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=patch_reliability,
                    cmap="RdYlGn", vmin=patch_reliability.min(),
                    vmax=patch_reliability.max(), s=8, alpha=0.8, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Reliability Score (relative)", fontsize=9)
    # Annotate the actual score range so the reader knows the absolute scale
    rmin, rmax = patch_reliability.min(), patch_reliability.max()
    ax.text(0.02, 0.02, f"score range: [{rmin:.3f}, {rmax:.3f}]",
            transform=ax.transAxes, fontsize=7, color="#555", va="bottom")
    ax.set_title("Continuous Reliability Score")
    ax.set_xlabel(f"{method_label} Dim 1"); ax.set_ylabel(f"{method_label} Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "phase2a_difficulty_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Phase 2b — Impossible-to-embed / outlier overlay ─────────────────────────

def plot_outlier_overlay(
    embedding: np.ndarray,
    lof_scores: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    lof_threshold: float,
    output_dir: str,
) -> str:
    """
    Two-panel figure on the baseline 2D embedding:
      Left:  normal points (grey) + outlier points (red, scaled by LOF score)
      Right: LOF score distribution histogram with threshold line
    """
    outlier_mask = lof_scores > lof_threshold
    n_outliers = outlier_mask.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Impossible-to-Embed Points  —  "
        f"{n_outliers} flagged ({100*n_outliers/len(lof_scores):.1f}%)",
        fontsize=13, fontweight="bold",
    )

    # ── Left: scatter with outliers ──
    ax = axes[0]
    normal_mask = ~outlier_mask
    if labels is not None:
        unique_labels = np.unique(labels[normal_mask]) if normal_mask.any() else []
        for i, lab in enumerate(unique_labels):
            m = normal_mask & (labels == lab)
            name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
            ax.scatter(embedding[m, 0], embedding[m, 1], c=_PALETTE[i % len(_PALETTE)],
                       label=name, s=6, alpha=0.5, edgecolors="none")
    else:
        ax.scatter(embedding[normal_mask, 0], embedding[normal_mask, 1],
                   c="#BDC3C7", s=6, alpha=0.5, edgecolors="none", label="Normal")

    if outlier_mask.any():
        sizes = 20 + 80 * (lof_scores[outlier_mask] - lof_threshold) / max(lof_scores.max() - lof_threshold, 1e-9)
        ax.scatter(embedding[outlier_mask, 0], embedding[outlier_mask, 1],
                   c="#E74C3C", s=np.clip(sizes, 15, 120), alpha=0.9,
                   edgecolors="#8B0000", linewidths=0.5, label=f"Outlier (LOF>{lof_threshold:.1f})")

    ax.legend(fontsize=8, markerscale=2, framealpha=0.85)
    ax.set_title("Outlier Points on Baseline Embedding")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Right: LOF histogram ──
    ax = axes[1]
    ax.hist(lof_scores, bins=50, color="#2E86AB", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(lof_threshold, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Threshold = {lof_threshold:.1f}")
    ax.set_xlabel("LOF Score")
    ax.set_ylabel("Count")
    ax.set_title("LOF Score Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "phase2b_outlier_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Phase 4 — Convergence curve ──────────────────────────────────────────────

def plot_convergence_curve(history: list, output_dir: str, quality_target: float = 8.0) -> str:
    """
    Multi-panel figure tracking quality scores and metrics across iterations.
      Top:    evaluator score, implicit score, combined score
      Bottom: key metrics (Spearman, Trustworthiness, Continuity, Silhouette, Stress)
    """
    iters = [h["iteration"] for h in history]
    eval_scores = [h.get("evaluator_score", 0) for h in history]
    impl_scores = [h.get("implicit_score", 0) for h in history]
    comb_scores = [h.get("combined_score", 0) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Optimisation Convergence", fontsize=13, fontweight="bold")

    # ── Top: quality scores ──
    ax = axes[0]
    ax.plot(iters, eval_scores, "o-", color="#2E86AB", linewidth=2, label="Evaluator (Agent 7)", markersize=6)
    ax.plot(iters, impl_scores, "s-", color="#27AE60", linewidth=2, label="Implicit (Agent 6)", markersize=6)
    ax.plot(iters, comb_scores, "^-", color="#E74C3C", linewidth=2.5, label="Combined", markersize=7)
    ax.axhline(quality_target, color="#F39C12", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Target ({quality_target:.1f})")
    ax.set_ylabel("Quality Score (0–10)")
    ax.set_title("Quality Scores per Iteration")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 10.5)

    # ── Bottom: reliability metrics ──
    ax = axes[1]
    metric_map = {
        "Trustworthiness": ("#27AE60", "o"),
        "Continuity":      ("#2E86AB", "s"),
        "Silhouette":      ("#8E44AD", "^"),
        "Spearman":        ("#F39C12", "D"),
    }
    for key, (col, marker) in metric_map.items():
        vals = [h.get("metrics_snapshot", {}).get(key) for h in history]
        if any(v is not None for v in vals):
            clean_iters = [i for i, v in zip(iters, vals) if v is not None]
            clean_vals  = [v for v in vals if v is not None]
            ax.plot(clean_iters, clean_vals, f"{marker}-", color=col, linewidth=1.8,
                    label=key, markersize=5)

    # Stress on secondary y-axis (different scale)
    stress_vals = [h.get("metrics_snapshot", {}).get("Stress") for h in history]
    if any(v is not None for v in stress_vals):
        ax2 = ax.twinx()
        clean_iters = [i for i, v in zip(iters, stress_vals) if v is not None]
        clean_vals  = [v for v in stress_vals if v is not None]
        ax2.plot(clean_iters, clean_vals, "x--", color="#E74C3C", linewidth=1.5,
                 label="Stress (right)", markersize=6)
        ax2.set_ylabel("Stress", color="#E74C3C", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#E74C3C")
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric Value")
    ax.set_title("Reliability Metrics per Iteration")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(iters)

    fig.tight_layout()
    path = os.path.join(output_dir, "phase4_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Phase 6 — Visual Forensics
# ═══════════════════════════════════════════════════════════════════════

def plot_distortion_heatmap(
    embedding: np.ndarray,
    composite_distortion: np.ndarray,
    pointwise_trust: np.ndarray,
    pointwise_cont: np.ndarray,
    pointwise_stress: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    output_dir: str,
) -> str:
    """
    Multi-panel distortion heatmap over the final embedding.

    Panel 1: Composite distortion heatmap with class label annotations
    Panel 2: Pointwise trustworthiness (local precision)
    Panel 3: Pointwise continuity (local recall)
    Panel 4: Pointwise stress (distance distortion)

    All panels overlay the data points with class labels annotated so the
    user can see which points / areas are highly distorted.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(
        "Visual Forensics — Per-Point Distortion Analysis",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Shared helper: annotate class centroids with labels
    def _annotate_centroids(ax, emb, labs, lnames):
        if labs is None:
            return
        for lab in np.unique(labs):
            mask = labs == lab
            cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
            name = lnames.get(int(lab), str(lab)) if lnames else str(lab)
            ax.annotate(
                name, (cx, cy), fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                          alpha=0.85, linewidth=0.5),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    panels = [
        ("Composite Distortion", composite_distortion, "hot_r",
         "Distortion (0=perfect, 1=worst)"),
        ("Pointwise Trustworthiness", pointwise_trust, "RdYlGn",
         "Trustworthiness (0=false neighbours, 1=perfect)"),
        ("Pointwise Continuity", pointwise_cont, "RdYlGn",
         "Continuity (0=lost neighbours, 1=perfect)"),
        ("Pointwise Stress", pointwise_stress, "hot_r",
         "Local Stress (higher=more distorted)"),
    ]

    for ax, (title, values, cmap, cbar_label) in zip(axes.flat, panels):
        # Plot all points coloured by metric
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=values, cmap=cmap, s=10, alpha=0.85, edgecolors="none",
            vmin=np.percentile(values, 2), vmax=np.percentile(values, 98),
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)

        # Annotate centroids
        _annotate_centroids(ax, embedding, labels, label_names)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.grid(True, linestyle="--", alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "phase6_distortion_heatmap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_highlighted_areas(
    embedding: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    highlighted_regions: List[Dict],
    composite_distortion: np.ndarray,
    output_dir: str,
) -> str:
    """
    Final annotated plot highlighting regions of interest identified by the
    Visual Forensics Evaluator agent.

    Each region in `highlighted_regions` is a dict with:
      - label: str (region name / description)
      - center_x, center_y: float (2D coordinates)
      - radius: float (approximate extent)
      - category: "high_distortion" | "well_preserved" | "artifact" | "interesting"
      - description: str

    Layout:
      Left:  Main scatter with class colours + highlighted region circles + labels
      Right: Distortion heatmap with highlighted region overlays
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(
        "Visual Forensics — Highlighted Regions of Interest",
        fontsize=14, fontweight="bold",
    )

    # Colour scheme per region category
    cat_colours = {
        "high_distortion": ("#E74C3C", "--"),
        "artifact":        ("#E74C3C", "-."),
        "well_preserved":  ("#27AE60", "-"),
        "interesting":     ("#2E86AB", ":"),
    }

    for panel_idx, ax in enumerate(axes):
        # Panel 0: class-coloured scatter; Panel 1: distortion heatmap
        if panel_idx == 0:
            if labels is not None:
                unique_labels = np.unique(labels)
                for i, lab in enumerate(unique_labels):
                    mask = labels == lab
                    name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
                    ax.scatter(
                        embedding[mask, 0], embedding[mask, 1],
                        c=_PALETTE[i % len(_PALETTE)],
                        label=f"{name} ({mask.sum()})",
                        s=8, alpha=0.65, edgecolors="none",
                    )
                ax.legend(
                    fontsize=7, markerscale=2, framealpha=0.85,
                    loc="upper right", ncol=max(1, len(unique_labels) // 5),
                )
            else:
                ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    c="#2E86AB", s=8, alpha=0.6, edgecolors="none",
                )
            ax.set_title("Final Embedding — Highlighted Regions", fontsize=11, fontweight="bold")
        else:
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=composite_distortion, cmap="hot_r", s=10, alpha=0.8,
                edgecolors="none",
                vmin=np.percentile(composite_distortion, 2),
                vmax=np.percentile(composite_distortion, 98),
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Composite Distortion", fontsize=8)

            # Annotate class centroids on distortion panel
            if labels is not None:
                for lab in np.unique(labels):
                    mask = labels == lab
                    cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
                    name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
                    ax.annotate(
                        name, (cx, cy), fontsize=7, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="gray", alpha=0.85, linewidth=0.4),
                    )
            ax.set_title("Distortion Heatmap — Highlighted Regions", fontsize=11, fontweight="bold")

        # Overlay highlighted regions on both panels
        for region in highlighted_regions:
            cx = region.get("center_x", 0)
            cy = region.get("center_y", 0)
            r = region.get("radius", 5)
            cat = region.get("category", "interesting")
            desc = region.get("label", "")
            colour, ls = cat_colours.get(cat, ("#2E86AB", ":"))

            circle = plt.Circle(
                (cx, cy), r, fill=False,
                edgecolor=colour, linewidth=2.0, linestyle=ls, alpha=0.9,
            )
            ax.add_patch(circle)

            # Label the region
            ax.annotate(
                desc, (cx, cy + r + 0.5),
                fontsize=7, fontweight="bold", color=colour,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour,
                          alpha=0.9, linewidth=0.6),
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.8),
            )

        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.grid(True, linestyle="--", alpha=0.2)

    # Add legend for region categories
    from matplotlib.lines import Line2D
    cat_handles = []
    cats_used = set(r.get("category", "interesting") for r in highlighted_regions)
    for cat_name, (col, ls) in cat_colours.items():
        if cat_name in cats_used:
            cat_handles.append(
                Line2D([0], [0], color=col, linestyle=ls, linewidth=2,
                       label=cat_name.replace("_", " ").title())
            )
    if cat_handles:
        axes[0].legend(
            handles=cat_handles + (axes[0].get_legend().legend_handles if axes[0].get_legend() else []),
            fontsize=7, loc="lower left", framealpha=0.85,
        )

    fig.tight_layout()
    path = os.path.join(output_dir, "phase6_highlighted_areas.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distortion_summary(
    embedding: np.ndarray,
    composite_distortion: np.ndarray,
    pointwise_trust: np.ndarray,
    pointwise_cont: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    output_dir: str,
) -> str:
    """
    Per-class distortion summary: box plots + bar charts showing which
    classes are most and least faithfully represented.
    """
    if labels is None:
        return ""

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    names = [label_names.get(int(l), str(l)) if label_names else str(l) for l in unique_labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Visual Forensics — Per-Class Distortion Summary",
        fontsize=13, fontweight="bold",
    )

    # Collect per-class stats
    class_distortion = [composite_distortion[labels == l] for l in unique_labels]
    class_trust = [pointwise_trust[labels == l] for l in unique_labels]
    class_cont = [pointwise_cont[labels == l] for l in unique_labels]

    # ── Panel 1: Composite distortion box plot ──
    ax = axes[0]
    bp = ax.boxplot(class_distortion, labels=names, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
        patch.set_alpha(0.7)
    ax.set_ylabel("Composite Distortion")
    ax.set_title("Distortion by Class")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # ── Panel 2: Mean trustworthiness per class ──
    ax = axes[1]
    means_t = [d.mean() for d in class_trust]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_classes)]
    bars = ax.bar(names, means_t, color=colors, alpha=0.8)
    for bar, val in zip(bars, means_t):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Mean Pointwise Trustworthiness")
    ax.set_title("Local Precision by Class")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # ── Panel 3: Mean continuity per class ──
    ax = axes[2]
    means_c = [d.mean() for d in class_cont]
    bars = ax.bar(names, means_c, color=colors, alpha=0.8)
    for bar, val in zip(bars, means_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Mean Pointwise Continuity")
    ax.set_title("Local Recall by Class")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "phase6_class_distortion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Phase 6 — Hierarchy Forensics: Dendrogram Comparison Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_dendrogram_comparison(
    dendrograms: dict,
    hierarchy_diff: dict,
    label_names: Optional[dict] = None,
    output_dir: str = OUTPUT_ROOT,
) -> str:
    """
    Side-by-side dendrograms (HD vs 2D) with topology changes highlighted.

    Layout: 3 panels
      Left:  HD dendrogram (UPGMA on class centroids)
      Centre: 2D dendrogram (UPGMA on 2D class centroids)
      Right: Topology change summary table

    Classes whose nearest hierarchical neighbour changed between HD and 2D
    are highlighted in red on both dendrograms.
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    hd = dendrograms.get("hierarchy_hd", {})
    ld = dendrograms.get("hierarchy_2d", {})
    Z_hd = hd.get("linkage")
    Z_ld = ld.get("linkage")
    hd_leaf_labels = hd.get("leaf_labels", [])
    ld_leaf_labels = ld.get("leaf_labels", [])

    if Z_hd is None or Z_ld is None:
        return ""

    # Resolve label names for display
    def _resolve(lab_str):
        if label_names:
            try:
                return label_names.get(int(lab_str), lab_str)
            except (ValueError, TypeError):
                return lab_str
        return lab_str

    hd_display = [_resolve(l) for l in hd_leaf_labels]
    ld_display = [_resolve(l) for l in ld_leaf_labels]

    # Identify affected classes from topology changes
    topo_changes = hierarchy_diff.get("topology_changes", [])
    affected_classes = set()
    for tc in topo_changes:
        if tc.get("type") == "nearest_neighbour_changed":
            affected_classes.add(tc.get("class", ""))
        elif tc.get("type") == "closest_pair_changed":
            for c in tc.get("hd_closest", []) + tc.get("ld_closest", []):
                affected_classes.add(str(c))
        elif tc.get("type") == "merge_order_inversion":
            for c in tc.get("classes", []):
                affected_classes.add(str(c))

    # Map affected classes back to their raw label strings for colouring
    affected_raw = set()
    for lab_str in hd_leaf_labels:
        if lab_str in affected_classes or _resolve(lab_str) in affected_classes:
            affected_raw.add(lab_str)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    fig.suptitle(
        "Visual Forensics — Hierarchy Comparison (HD vs 2D)",
        fontsize=14, fontweight="bold",
    )

    # ── Left: HD dendrogram ──
    ax = axes[0]
    scipy_dendrogram(
        Z_hd, labels=hd_display, ax=ax, orientation="top",
        leaf_rotation=45, leaf_font_size=9,
        above_threshold_color="#BDC3C7",
    )
    ax.set_title("High-Dimensional (HD) Hierarchy", fontsize=11, fontweight="bold")
    ax.set_ylabel("UPGMA Distance")
    # Highlight affected leaf labels
    for lbl in ax.get_xticklabels():
        raw_match = False
        for lab_str in affected_raw:
            if _resolve(lab_str) == lbl.get_text():
                raw_match = True
                break
        if raw_match:
            lbl.set_color("#E74C3C")
            lbl.set_fontweight("bold")

    # ── Centre: 2D dendrogram ──
    ax = axes[1]
    scipy_dendrogram(
        Z_ld, labels=ld_display, ax=ax, orientation="top",
        leaf_rotation=45, leaf_font_size=9,
        above_threshold_color="#BDC3C7",
    )
    ax.set_title("2D Embedding Hierarchy", fontsize=11, fontweight="bold")
    ax.set_ylabel("UPGMA Distance")
    for lbl in ax.get_xticklabels():
        raw_match = False
        for lab_str in affected_raw:
            if _resolve(lab_str) == lbl.get_text():
                raw_match = True
                break
        if raw_match:
            lbl.set_color("#E74C3C")
            lbl.set_fontweight("bold")

    # ── Right: topology change summary ──
    ax = axes[2]
    ax.axis("off")
    ax.set_title("Topology Changes", fontsize=11, fontweight="bold")

    rf = hierarchy_diff.get("robinson_foulds_distance", "?")
    coph = hierarchy_diff.get("cophenetic_correlation", "?")
    text_lines = [
        f"Robinson-Foulds: {rf}",
        f"Cophenetic corr: {coph}",
        "",
    ]
    if topo_changes:
        for tc in topo_changes[:10]:
            sev = tc.get("severity", "?")
            icon = "[!]" if sev == "high" else "[~]" if sev == "medium" else "[.]"
            desc = tc.get("description", "")
            if len(desc) > 60:
                desc = desc[:57] + "..."
            text_lines.append(f"{icon} {desc}")
    else:
        text_lines.append("[OK] No significant topology changes.")

    ax.text(
        0.05, 0.95, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9FA", ec="#DEE2E6", alpha=0.9),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "phase6_dendrogram_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hierarchy_change_detail(
    dendrograms: dict,
    hierarchy_diff: dict,
    embedding: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[dict],
    output_dir: str = OUTPUT_ROOT,
) -> str:
    """
    Detailed hierarchy change plot: 2D scatter with arrows showing which
    class nearest-neighbour relationships changed between HD and 2D, plus
    a merge-rank comparison lollipop chart.
    """
    if labels is None:
        return ""

    topo_changes = hierarchy_diff.get("topology_changes", [])
    nn_changes = [tc for tc in topo_changes if tc.get("type") == "nearest_neighbour_changed"]

    if not nn_changes and not topo_changes:
        return ""

    unique_labels = np.unique(labels)
    centroids = {}
    for lab in unique_labels:
        mask = labels == lab
        name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
        centroids[name] = (float(embedding[mask, 0].mean()), float(embedding[mask, 1].mean()))

    def _resolve(lab_str):
        if label_names:
            try:
                return label_names.get(int(lab_str), lab_str)
            except (ValueError, TypeError):
                return lab_str
        return lab_str

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(
        "Visual Forensics — Hierarchy Change Detail",
        fontsize=14, fontweight="bold",
    )

    # ── Left: 2D scatter with hierarchy arrows ──
    ax = axes[0]
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=_PALETTE[i % len(_PALETTE)],
            label=f"{name} ({mask.sum()})",
            s=8, alpha=0.5, edgecolors="none",
        )

    for tc in nn_changes:
        cls_name = _resolve(tc.get("class", ""))
        hd_nn = _resolve(tc.get("hd_nearest", ""))
        ld_nn = _resolve(tc.get("ld_nearest", ""))

        if cls_name in centroids:
            cx, cy = centroids[cls_name]
            if hd_nn in centroids:
                hx, hy = centroids[hd_nn]
                ax.annotate(
                    "", xy=(hx, hy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#27AE60", linewidth=2,
                                    linestyle="-"),
                )
            if ld_nn in centroids:
                lx, ly = centroids[ld_nn]
                ax.annotate(
                    "", xy=(lx, ly), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", linewidth=2,
                                    linestyle="--"),
                )

    for name, (cx, cy) in centroids.items():
        is_affected = any(
            _resolve(tc.get("class", "")) == name for tc in nn_changes
        )
        ax.annotate(
            name, (cx, cy), fontsize=8,
            fontweight="bold" if is_affected else "normal",
            color="#E74C3C" if is_affected else "#2C3E50",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                      alpha=0.85, linewidth=0.5),
        )

    from matplotlib.lines import Line2D
    arrow_handles = [
        Line2D([0], [0], color="#27AE60", linewidth=2, linestyle="-",
               label="HD nearest neighbour"),
        Line2D([0], [0], color="#E74C3C", linewidth=2, linestyle="--",
               label="2D nearest neighbour"),
    ]
    ax.legend(handles=arrow_handles, fontsize=8, loc="lower right", framealpha=0.85)
    ax.set_title("Nearest-Neighbour Hierarchy Changes", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.2)

    # ── Right: merge-rank comparison ──
    ax = axes[1]
    merge_inversions = [tc for tc in topo_changes if tc.get("type") == "merge_order_inversion"]
    if merge_inversions:
        inversions_sorted = sorted(
            merge_inversions,
            key=lambda x: abs(x.get("ld_merge_rank", 0) - x.get("hd_merge_rank", 0)),
            reverse=True,
        )[:12]
        y_labels_chart = []
        hd_ranks = []
        ld_ranks = []
        for inv in inversions_sorted:
            classes = inv.get("classes", ["?", "?"])
            y_labels_chart.append(f"{classes[0]}↔{classes[1]}")
            hd_ranks.append(inv.get("hd_merge_rank", 0))
            ld_ranks.append(inv.get("ld_merge_rank", 0))

        y_pos = np.arange(len(y_labels_chart))
        ax.barh(y_pos - 0.15, hd_ranks, height=0.3, color="#27AE60", alpha=0.8, label="HD rank")
        ax.barh(y_pos + 0.15, ld_ranks, height=0.3, color="#E74C3C", alpha=0.8, label="2D rank")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels_chart, fontsize=8)
        ax.set_xlabel("Merge Rank (lower = merged earlier)")
        ax.set_title("Merge-Order Inversions", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No merge-order inversions detected",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("Merge-Order Inversions", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.2, axis="x")

    fig.tight_layout()
    path = os.path.join(output_dir, "phase6_hierarchy_changes.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
