#!/usr/bin/env python3
"""
main.py — CLI entry point for AutoManifold (Optimization mode).

Usage:
    python main.py --data data.csv --labels labels.csv \\
        --provider openai --model gpt-5.4 --api-key $OPENAI_API_KEY \\
        --dr-method tsne --output outputs/

See `python main.py --help` for full option list.
"""

from __future__ import annotations

import os

# Force Numba's threading layer to a fork-safe backend BEFORE any Numba-backed
# import (umap, pacmap). Avoids TBB "fork from non-main thread" warnings.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

# Load API keys from a project-local .env if present (shell exports still win).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
except Exception:
    pass

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from pipeline.config import (
    OUTPUT_ROOT,
    LLMConfig,
    ConvergenceConfig,
    PatchConfig,
    MetricsConfig,
    PipelineConfig,
)
from pipeline.planner import Planner
from pipeline.datasets import DATASETS, DatasetUnavailableError, get_dataset, list_datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic DR Algorithm Design Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data — either --dataset (built-in) or --data / --labels (files) ──
    p.add_argument("--dataset", default=None,
                   choices=list(DATASETS.keys()),
                   metavar="DATASET",
                   help="Built-in bio/chem dataset key. Run --list-datasets to see options.")
    p.add_argument("--list-datasets", action="store_true",
                   help="Print available built-in datasets and exit.")
    p.add_argument("--data", default=None, help="Path to data matrix (CSV or .npy)")
    p.add_argument("--labels", default=None, help="Path to label vector (CSV or .npy)")
    p.add_argument("--label-names", default=None, help="JSON file mapping int→name")

    # ── LLM ──
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic", "google", "deepseek"])
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--api-key", default=None, help="API key (or set env var)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Max output tokens (default: no limit, model decides)")

    # ── DR method ──
    p.add_argument("--dr-method", default="tsne", choices=["tsne", "umap", "pacmap", "pca"])
    p.add_argument("--primary-goal", default="balance both",
                   help="'local structure', 'global structure', or 'balance both'")

    # ── t-SNE hyperparameters ──
    p.add_argument("--n-pcs", type=int, default=20)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-lr", type=float, default=200.0)
    p.add_argument("--tsne-max-iter", type=int, default=1000)
    p.add_argument("--tsne-early-exaggeration", type=float, default=12.0)
    p.add_argument("--tsne-angle", type=float, default=0.5)
    p.add_argument("--tsne-init", default="pca", choices=["pca", "random"])
    p.add_argument("--tsne-metric", default="euclidean")
    p.add_argument("--tsne-n-iter-without-progress", type=int, default=300)

    # ── UMAP hyperparameters ──
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", default="euclidean")
    p.add_argument("--umap-n-epochs", type=int, default=0,
                   help="0 = UMAP auto-selects")
    p.add_argument("--umap-lr", type=float, default=1.0,
                   dest="umap_learning_rate")
    p.add_argument("--umap-spread", type=float, default=1.0)
    p.add_argument("--umap-negative-sample-rate", type=int, default=5)
    p.add_argument("--umap-repulsion-strength", type=float, default=1.0)
    p.add_argument("--umap-set-op-mix-ratio", type=float, default=1.0)
    p.add_argument("--umap-local-connectivity", type=float, default=1.0)

    # ── PaCMAP hyperparameters ──
    p.add_argument("--pacmap-n-neighbors", type=int, default=10)
    p.add_argument("--pacmap-mn-ratio", type=float, default=0.5)
    p.add_argument("--pacmap-fp-ratio", type=float, default=2.0)
    p.add_argument("--pacmap-lr", type=float, default=0.01)
    p.add_argument("--pacmap-iters-p1", type=int, default=100,
                   dest="pacmap_num_iters_phase1")
    p.add_argument("--pacmap-iters-p2", type=int, default=100,
                   dest="pacmap_num_iters_phase2")
    p.add_argument("--pacmap-iters-p3", type=int, default=250,
                   dest="pacmap_num_iters_phase3")
    p.add_argument("--pacmap-no-pca", action="store_false", dest="pacmap_apply_pca",
                   help="Disable PaCMAP's internal PCA preprocessing")
    p.add_argument("--pacmap-distance", default="euclidean")

    # ── Convergence ──
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--quality-target", type=float, default=8.0)
    p.add_argument("--min-improvement-threshold", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)

    # ── Metrics ──
    p.add_argument("--neighborhood-k", type=int, default=10)
    p.add_argument("--lof-neighbors", type=int, default=20)

    # ── Patch ──
    p.add_argument("--patch-size", type=int, default=100)

    # ── Output ──
    p.add_argument("--output", default=OUTPUT_ROOT, help="Base output directory")
    p.add_argument("--experiment", default=None,
                   help="Experiment name (subfolder under --output). Defaults to a timestamp.")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def load_data(path: str) -> np.ndarray:
    """Load feature matrix — drops non-numeric, all-NaN columns and NaN rows."""
    if path.endswith(".npy"):
        return np.load(path)
    df = pd.read_csv(path)
    # Drop obvious ID columns
    id_cols = [c for c in df.columns if c.lower() in ("id", "sample_id", "patient_id", "index")]
    df = df.drop(columns=id_cols, errors="ignore")
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in {path}")
    # Drop columns that are entirely NaN (e.g. trailing-comma artefacts)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    # Drop rows with any remaining NaN
    numeric_df = numeric_df.dropna(axis=0, how="any")
    return numeric_df.values.astype(np.float64)


def load_labels(path: str) -> np.ndarray:
    """Load label vector — uses the first non-numeric / categorical column."""
    if path.endswith(".npy"):
        return np.load(path)
    df = pd.read_csv(path)
    # Drop obvious ID columns
    id_cols = [c for c in df.columns if c.lower() in ("id", "sample_id", "patient_id", "index")]
    df = df.drop(columns=id_cols, errors="ignore")
    # Prefer the first non-numeric column as the label
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        col = non_numeric[0]
        series = df[col].astype(str)
    else:
        # Fall back to the first numeric column
        col = df.columns[0]
        series = df[col].astype(str)
    # Encode to integers
    unique = sorted(series.unique())
    mapping = {name: i for i, name in enumerate(unique)}
    return np.array([mapping[v] for v in series], dtype=int)


def main():
    args = parse_args()

    # ── Early exits ──
    if args.list_datasets:
        list_datasets()
        sys.exit(0)

    if not args.dataset and not args.data:
        print("ERROR: Provide either --dataset <key> or --data <path>.", file=sys.stderr)
        print("       Run with --list-datasets to see available built-in datasets.", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Resolve API key ──
    api_key = args.api_key or os.environ.get(
        {"openai": "OPENAI_API_KEY",
         "anthropic": "ANTHROPIC_API_KEY",
         "google": "GEMINI_API_KEY",
         "deepseek": "DEEPSEEK_API_KEY"}[args.provider]
    )
    if not api_key:
        print(f"ERROR: No API key provided for {args.provider}. "
              f"Pass --api-key or set the environment variable.", file=sys.stderr)
        sys.exit(1)

    # ── Build config ──
    config = PipelineConfig(
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        convergence=ConvergenceConfig(
            max_iterations=args.max_iter,
            quality_score_target=args.quality_target,
            min_improvement_threshold=args.min_improvement_threshold,
            patience=args.patience,
        ),
        patch=PatchConfig(target_patch_size=args.patch_size),
        metrics=MetricsConfig(
            neighborhood_k=args.neighborhood_k,
            lof_neighbors=args.lof_neighbors,
        ),
        default_dr_method=args.dr_method,
        primary_goal=args.primary_goal,
        n_pcs=args.n_pcs,
        # t-SNE
        tsne_perplexity=args.tsne_perplexity,
        tsne_learning_rate=args.tsne_lr,
        tsne_max_iter=args.tsne_max_iter,
        tsne_early_exaggeration=args.tsne_early_exaggeration,
        tsne_angle=args.tsne_angle,
        tsne_init=args.tsne_init,
        tsne_metric=args.tsne_metric,
        tsne_n_iter_without_progress=args.tsne_n_iter_without_progress,
        # UMAP
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_n_epochs=args.umap_n_epochs,
        umap_learning_rate=args.umap_learning_rate,
        umap_spread=args.umap_spread,
        umap_negative_sample_rate=args.umap_negative_sample_rate,
        umap_repulsion_strength=args.umap_repulsion_strength,
        umap_set_op_mix_ratio=args.umap_set_op_mix_ratio,
        umap_local_connectivity=args.umap_local_connectivity,
        # PaCMAP
        pacmap_n_neighbors=args.pacmap_n_neighbors,
        pacmap_MN_ratio=args.pacmap_mn_ratio,
        pacmap_FP_ratio=args.pacmap_fp_ratio,
        pacmap_lr=args.pacmap_lr,
        pacmap_num_iters_phase1=args.pacmap_num_iters_phase1,
        pacmap_num_iters_phase2=args.pacmap_num_iters_phase2,
        pacmap_num_iters_phase3=args.pacmap_num_iters_phase3,
        pacmap_apply_pca=args.pacmap_apply_pca,
        pacmap_distance=args.pacmap_distance,
    )

    # ── Load data ──
    import datetime
    label_names = None

    if args.dataset:
        try:
            data, labels, ln_list, ds_info = get_dataset(args.dataset)
        except DatasetUnavailableError as exc:
            print(f"ERROR: {exc}\n       See the Datasets section of README.md.", file=sys.stderr)
            sys.exit(1)
        # Convert list → dict expected by planner
        label_names = {i: name for i, name in enumerate(ln_list)}
        logging.info("Dataset: %s  |  %s", ds_info["name"], ds_info["description"])
        logging.info("Shape: %s  |  Classes: %s", data.shape, ds_info["n_classes"])
        # default_exp = f"{args.dataset}_{args.dr_method}"
        default_exp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        data = load_data(args.data)
        labels = load_labels(args.labels) if args.labels else None
        if args.label_names:
            with open(args.label_names) as f:
                raw = json.load(f)
            label_names = {int(k): v for k, v in raw.items()}
        logging.info("Data: %s, Labels: %s", data.shape,
                     labels.shape if labels is not None else None)
        default_exp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Build experiment output dir ──
    exp_name = args.experiment or default_exp
    exp_dir = os.path.join(args.output, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logging.info("Experiment directory: %s", exp_dir)

    # ── Run pipeline ──
    planner = Planner(config)
    result = planner.run(data, labels=labels, label_names=label_names, output_dir=exp_dir)

    # ── Print summary ──
    best = result.get("best_result", {})
    m = best.get("metrics", {})
    g = m.get("Global Structure & HD<->2D Correlation", {})
    loc = m.get("Neighborhood Preservation", {})
    clust = m.get("Clustering & Label Agreement", {})
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Experiment:      {exp_dir}")
    print(f"  Best iteration:  {best.get('iteration')}")
    print(f"  Best hyperparams: {best.get('hyperparameters')}")
    trust_key = next((k for k in loc if k.startswith("Trustworthiness")), None)
    cont_key  = next((k for k in loc if k.startswith("Continuity")), None)
    if trust_key: print(f"  {trust_key}: {loc[trust_key]:.4f}")
    if cont_key:  print(f"  {cont_key}:       {loc[cont_key]:.4f}")
    sil = clust.get("Silhouette Score")
    if isinstance(sil, float): print(f"  Silhouette Score: {sil:.4f}")
    spear = g.get("Spearman Correlation")
    if isinstance(spear, float): print(f"  Spearman:         {spear:.4f}")
    print(f"  Plot:            {best.get('plot_path')}")
    print(f"  Full report:     {exp_dir}/final_report.json")
    print(f"  History:         {exp_dir}/optimization_history.json")


if __name__ == "__main__":
    main()
