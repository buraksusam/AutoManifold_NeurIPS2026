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

| Mode (key) | Planner | Description |
|------------|---------|-------------|
| **AutoManifold** (`automanifold`) | `AutoManifoldPlanner` | Specialised agents characterise the data, analyse embedding difficulty and neighbourhoods, design affinity and cost components from a primitive library, and synthesise, validate and refine a novel algorithm (DR_Template + DR_Template_v2) |
| **NoSkel-Agents** (`noskel_agents`) | `NoSkelAgentsPlanner` | Skeleton-free design: a designer agent builds the algorithm from a blank canvas, spawning analysis sub-agents on demand (DR_Template_v2 constraints, no primitive skeleton) |
| **Auto-Skills** (`auto_skills`) | `AutoSkillsPlanner` | One autonomous LLM agent drives the design by calling skills (`pipeline/skills/auto_skills/`) and deterministic compute tools |
| **SingleLLM** (`singlellm`) | `SingleLLMPlanner` | Baseline: a single blank-prompt LLM call produces the whole algorithm; no agents, no data context |
| **Optimization** (`optimization`) | `Planner` | Tune an existing method (t-SNE / UMAP / PaCMAP / PCA) with LLM-guided iterative refinement |

All algorithm-design modes share the same execution, evaluation (baseline
comparison, metrics) and visual-forensics stages, so their results are directly
comparable.

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

## Design documents

The agent prompts are derived from two design templates, included in `docs/`:

| Document | Defines | Used by |
|----------|---------|---------|
| [`docs/DR_Template.md`](docs/DR_Template.md) | Auxiliary analysis tasks (§1.1 difficulty levels, §1.2 maximal embeddable neighbourhoods, §1.3 patch stitching, §1.4 impossible-to-embed points) and implementation strategies (§2.1 data feeding, §2.2 fine-tuning, §2.3 hyperparameter optimisation) | Agents 1–3 and 6 (AutoManifold, NoSkel-Agents' data characterisation, Optimization); Agents 4–5 patch stitching / outliers (Optimization); Agent 8 strategy options; Auto-Skills skills `map-difficulty-to-primitives`, `estimate-neighborhoods` |
| [`docs/DR_Template_v2.md`](docs/DR_Template_v2.md) | Synthesis and design constraints: verified-code reuse, dynamic-code scope, score plan, hyperparameter-selection policy | Agent 8 synthesis (AutoManifold, NoSkel-Agents); NoSkel-Agents designer; Auto-Skills `synthesize-algorithm` skill |

Module docstrings point to the relevant section, e.g.
`pipeline/agents/agent_2_difficulty_analyzer.py` → DR_Template §1.1.

The hyperparameter-optimisation loop (Agent 6) follows the iterative
refinement methodology of *Explainable Iterative Data Visualisation Refinement
via an LLM Agent* (Susam & Mu, IEEE CG&A 2026). The Auto-Skills planner prompt
and skill definitions live in `pipeline/skills/auto_skills/` (`PLANNER.md`,
`*/SKILL.md`).

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
├── skills/        # Auto-Skills planner prompt + skill definitions
├── tools/         # tool registry for the Auto-Skills agent loop
├── planner_automanifold.py   # AutoManifold (base class for NoSkel-Agents, SingleLLM)
├── planner_noskel_agents.py  # NoSkel-Agents
├── planner_auto_skills.py    # Auto-Skills
├── planner_singlellm.py      # SingleLLM
├── planner.py     # Optimization
├── datasets.py    # dataset registry and loaders
└── config.py      # LLM / pipeline configuration
app.py             # FastAPI backend for the web UI
frontend/          # web UI
main.py            # CLI entry point
serve.py           # web UI launcher
docs/              # DR_Template.md, DR_Template_v2.md
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
