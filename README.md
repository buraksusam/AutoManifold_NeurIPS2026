# AutoManifold

Code for the NeurIPS 2026 paper **"AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding"**.

AutoManifold is a multi-agent LLM framework that designs manifold-embedding
algorithms for data visualisation, steered toward user-specified structural
priorities. It characterises a dataset, analyses its embedding difficulty, and
then designs, implements, evaluates and iteratively refines an embedding
algorithm for it.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add the API key(s) for your LLM provider
```

Supported LLM providers: OpenAI, Anthropic, Google (Gemini) and DeepSeek.
Keys are read from `.env` or from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`).

## Usage

### Web UI (all modes)

```bash
python serve.py --open          # http://localhost:8000
```

Choose a dataset (built-in or uploaded CSV), an LLM provider/model, an
optimisation goal and a pipeline mode:

| Mode | Description |
|------|-------------|
| `optimization` | Tune an existing method (t-SNE / UMAP / PaCMAP / PCA) with LLM-guided iterative refinement |
| `novel` | Generate a novel DR algorithm by composing algorithmic primitives (affinity, cost, optimiser) |
| `novelv2` | `novel` with DR_Template_v2 synthesis constraints |
| `novelv2_skills` | `novelv2` deterministic stages driven by a single LLM context via skills |
| `all_llm` | Unconstrained, blank-canvas algorithm design by the LLM |
| `allLLMv2` | `all_llm` with DR_Template_v2 design/synthesis constraints |
| `ablation` | Single-LLM blank prompt baseline (no agents) |

### Command line (optimisation mode)

```bash
python main.py --list-datasets
python main.py --dataset iris --provider openai --model gpt-5.4 --dr-method tsne
python main.py --data X.csv --labels y.csv --provider anthropic --model <model>
```

Run `python main.py --help` for all options.

### Outputs

Each run writes its artifacts (agent reports, plots, metrics, the synthesised
algorithm code, embeddings) to `outputs/<run_id>/`. Uploaded files go to
`uploads/`. Both locations can be changed with `AUTOMANIFOLD_OUTPUT_DIR` and
`AUTOMANIFOLD_UPLOAD_DIR`.

## Datasets

Most datasets (Iris, Wine, Digits, Breast Cancer, MNIST, Fashion-MNIST,
Olivetti, LFW, 20 Newsgroups, Mice Protein, synthetic sets) are downloaded
automatically via scikit-learn / OpenML on first use.

The scRNA-seq datasets must be placed under `data/` manually:

```
data/
├── pbmc3k_processed.h5ad          # scanpy.datasets.pbmc3k_processed()
├── aging_human_skin/   {exprMatrix.tsv.gz, meta.tsv}
├── cortex_development/ {exprMatrix.tsv.gz, meta.tsv}
├── mouse_incisor/      {exprMatrix.tsv.gz, meta.tsv}
└── scRNA-seq/          {exprMatrix.tsv.gz, meta.tsv}   # human kidney
```

`pbmc3k_processed.h5ad` can be obtained with
`python -c "import scanpy as sc; sc.datasets.pbmc3k_processed().write('data/pbmc3k_processed.h5ad')"`.
The four other datasets are exported from the
[UCSC Cell Browser](https://cells.ucsc.edu/) (`exprMatrix.tsv.gz` + `meta.tsv`
from each dataset's download page). Preprocessed caches are created
automatically on first load.

## Tests

```bash
pytest tests
```

The tests use mocked LLM clients and need no API keys.

## Repository structure

```
pipeline/
├── agents/        # specialised worker agents (data characterisation, difficulty,
│                  #   neighbourhoods, affinity/cost design, synthesis, evaluation, ...)
├── engines/       # DR, clustering and metrics engines
├── skills/        # skill definitions for the skills-driven mode
├── tools/         # tool registry for the agentic loop
├── planner*.py    # orchestrators for each pipeline mode
├── datasets.py    # dataset registry and loaders
└── config.py      # LLM / pipeline configuration
app.py             # FastAPI backend for the web UI
frontend/          # web UI
main.py            # CLI entry point
serve.py           # web UI launcher
tests/
```

## Citation

```bibtex
@inproceedings{automanifold2026,
  title     = {AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
