"""
app.py — Web frontend server for the DR Design Pipeline.

Run with:
    python app.py                    # default port 8000
    python app.py --port 8080

Open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import os

# Force Numba's threading layer to a fork-safe backend BEFORE any Numba-backed
# import (umap, pacmap). Avoids TBB "fork from non-main thread" warnings.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

# Load API keys from a project-local .env if present. We use override=False so
# anything already exported in the shell still wins.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except Exception:
    pass

import argparse
import asyncio
import importlib
import json
import logging
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import OUTPUT_ROOT, PROJECT_ROOT, UPLOAD_ROOT

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DR Design Pipeline")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Run registry  {run_id → RunState}
# ---------------------------------------------------------------------------

runs: Dict[str, Dict[str, Any]] = {}

# Pipeline modes: key -> (module, planner class, display name)
MODES: Dict[str, tuple] = {
    "optimization":  ("pipeline.planner",               "Planner",             "Optimization"),
    "automanifold":  ("pipeline.planner_automanifold",  "AutoManifoldPlanner", "AutoManifold"),
    "noskel_agents": ("pipeline.planner_noskel_agents", "NoSkelAgentsPlanner", "NoSkel-Agents"),
    "auto_skills":   ("pipeline.planner_auto_skills",   "AutoSkillsPlanner",   "Auto-Skills"),
    "singlellm":     ("pipeline.planner_singlellm",     "SingleLLMPlanner",    "SingleLLM"),
}


def _make_run_id(experiment_name: Optional[str]) -> str:
    """Create a filesystem-safe run identifier.

    If no experiment name is provided, fall back to the current datetime format.
    If the chosen name already exists, append a timestamp suffix to keep it unique.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_name = (experiment_name or "").strip()
    if not raw_name:
        return timestamp

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
    safe_name = safe_name.strip("._-")
    if not safe_name:
        return timestamp

    candidate = safe_name
    if candidate in runs or os.path.exists(os.path.join(OUTPUT_ROOT, candidate)):
        candidate = f"{safe_name}_{timestamp}"
    return candidate


# ---------------------------------------------------------------------------
# Stdout capture — redirects pipeline thread's print() into a queue
# ---------------------------------------------------------------------------

class _StreamToQueue:
    """sys.stdout replacement that tees to the real stdout AND a queue."""

    def __init__(self, q: queue.Queue, original):
        self._q = q
        self._orig = original
        self._buf = ""

    def write(self, text: str) -> None:
        if self._orig:
            self._orig.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put({"type": "log", "data": {"line": stripped, "level": "info"}})

    def flush(self) -> None:
        if self._orig:
            self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno() if self._orig else 1


class _QueueLogHandler(logging.Handler):
    """logging.Handler that puts formatted records into a queue."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.setFormatter(logging.Formatter("%(levelname)s  %(name)s — %(message)s"))
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname.lower()
            self._q.put({"type": "log", "data": {"line": self.format(record), "level": level}})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pipeline runner  (background thread)
# ---------------------------------------------------------------------------

def _run_pipeline(
    run_id: str,
    cfg: dict,
    data: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[Dict[int, str]],
    output_dir: str,
) -> None:
    """Runs in a daemon thread; writes structured events into runs[run_id]['q'].

    Modes (see MODES):
    - 'optimization':  Planner — LLM-guided hyperparameter tuning of t-SNE/UMAP/PaCMAP/PCA
    - 'automanifold':  AutoManifoldPlanner — template-based composition of primitives
    - 'noskel_agents': NoSkelAgentsPlanner — skeleton-free multi-agent design
    - 'auto_skills':   AutoSkillsPlanner — autonomous agent loop over skills
    - 'singlellm':     SingleLLMPlanner — single blank-prompt LLM baseline
    """
    run = runs[run_id]
    q: queue.Queue = run["q"]
    mode = cfg.get("mode", "optimization")  # default to optimization mode

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    stream = _StreamToQueue(q, orig_stdout)
    sys.stdout = stream
    sys.stderr = stream

    log_handler = _QueueLogHandler(q)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        q.put({"type": "status", "data": {"status": "running"}})

        # Start skill tracking
        from pipeline.skill_tracker import get_tracker
        tracker = get_tracker()
        tracker.start_run(run_id)

        # Imports here so failures surface as proper events
        from pipeline.config import (
            ConvergenceConfig, LLMConfig, MetricsConfig, PatchConfig, PipelineConfig,
        )

        # Import the appropriate planner based on mode
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODES)}")
        module_name, class_name, display_name = MODES[mode]
        PlannerClass = getattr(importlib.import_module(module_name), class_name)
        q.put({"type": "log", "data": {"line": f"Using {display_name} planner", "level": "info"}})

        planner = PlannerClass(config)

        # Auto-Skills: stream per-skill-turn breadcrumbs to the UI log
        if mode == "auto_skills":
            def _skill_log(line: str, level: str = "info"):
                q.put({"type": "log", "data": {"line": line, "level": level}})
            planner.skill_log_callback = _skill_log

            # Per-token LLM stream → dedicated SSE event so the frontend
            # renders chunks into a streaming pane (not the line-oriented log).
            def _skill_stream(chunk: str):
                q.put({"type": "skill_stream", "data": {"chunk": chunk}})
            planner.skill_stream_callback = _skill_stream

            # Reasoning/thinking-content stream — emitted BEFORE the final
            # answer when the model has extended-thinking enabled.
            def _skill_thinking(chunk: str):
                q.put({"type": "skill_thinking", "data": {"chunk": chunk}})
            planner.skill_thinking_callback = _skill_thinking

            # Agent-loop tool-call lifecycle — fires twice per dispatch:
            #   ("call",   {"tool_name", "arguments", "iteration"})
            #   ("result", {"tool_name", "result_summary", "iteration"})
            def _tool_call(event_type: str, payload: dict):
                """Generic dispatcher for tool-call lifecycle events."""
                q.put({"type": "tool_call", "data": {"event": event_type, **payload}})
            planner.tool_call_callback = _tool_call

        result = planner.run(data, labels=labels, label_names=label_names, output_dir=output_dir)

        # End skill tracking and save log
        skill_log_path = tracker.end_run()
        if skill_log_path:
            q.put({"type": "skill_log", "data": {
                "path": skill_log_path,
                "run_id": run_id,
            }})

        run["status"] = "complete"
        run["result"] = result
        run["skill_log_path"] = skill_log_path
        q.put({"type": "complete", "data": {"status": "success", "output_dir": output_dir}})

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        run["status"] = "error"
        run["error"] = str(exc)
        q.put({"type": "error", "data": {"message": str(exc), "traceback": tb}})

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        root_logger.removeHandler(log_handler)
        q.put({"type": "_sentinel"})


# ---------------------------------------------------------------------------
# File watcher helpers
# ---------------------------------------------------------------------------

def _check_new_files(output_dir: str, seen: set, run_id: str) -> List[dict]:
    events = []
    if not os.path.isdir(output_dir):
        return events

    for fname in sorted(os.listdir(output_dir)):
        if fname in seen:
            continue
        seen.add(fname)
        fpath = os.path.join(output_dir, fname)
        url = f"/files/{run_id}/{fname}"

        # --- Iteration embedding plot ---
        m = re.match(r"iter_(\d+)\.png$", fname)
        if m:
            events.append({"type": "plot", "data": {"iteration": int(m.group(1)), "url": url}})
            continue

        # --- Per-iteration metrics JSON ---
        m = re.match(r"iter_(\d+)_metrics\.json$", fname)
        if m:
            try:
                with open(fpath) as f:
                    events.append({"type": "metrics", "data": json.load(f)})
            except Exception:
                pass
            continue

        # --- Agent 6 optimization JSON ---
        m = re.match(r"iter_(\d+)_agent6_optimization\.json$", fname)
        if m:
            try:
                with open(fpath) as f:
                    events.append({"type": "agent6", "data": {
                        "iteration": int(m.group(1)), "result": json.load(f)}})
            except Exception:
                pass
            continue

        # --- Agent 7 evaluation JSON ---
        m = re.match(r"iter_(\d+)_agent7_evaluation\.json$", fname)
        if m:
            try:
                with open(fpath) as f:
                    events.append({"type": "agent7", "data": {
                        "iteration": int(m.group(1)), "result": json.load(f)}})
            except Exception:
                pass
            continue

        # --- Named agent JSON files ---
        # Standard optimization planner agents (hyperparameter-only mode)
        AGENT_MAP_OPT = {
            "agent1_data_characterization.json": ("1", "Data Characterisation"),
            "agent2_difficulty_classification.json": ("2", "Difficulty Analysis"),
            "agent3_maximal_neighborhoods.json": ("3", "Neighborhood Analysis"),
            "agent4_patch_stitching.json": ("4", "Patch Stitching"),
            "agent5_outlier_identification.json": ("5", "Outlier Identification"),
            "agent9_visual_forensics.json": ("9", "Visual Forensics"),
            "default_vs_best_comparison.json": ("cmp", "Default vs Best Comparison"),
            "optimization_history.json": ("history", ""),
            "final_report.json": ("report", "Final Report"),
        }
        # AutoManifold planner agents
        AGENT_MAP_NOVEL = {
            "agent1_data_characterization.json": ("1", "Data Characterisation"),
            "agent2_difficulty_primitive_mapping.json": ("2", "Difficulty & Primitive Mapping"),
            "agent3_maximal_neighborhoods.json": ("3", "Neighborhood Analysis"),
            "agent4_affinity_design.json": ("4A", "Affinity Graph Design"),
            "agent5_cost_composition.json": ("5C", "Cost Function Composition"),
            "agent12_novelty_validation.json": ("12", "Novelty Validation"),
            "agent8_final_synthesis.json": ("8f", "Final Algorithm Synthesis"),
            "novel_metrics.json": ("novel_metrics", "Novel Algorithm Metrics"),
            "novel_comparison.json": ("novel_cmp", "Novel vs Baselines Comparison"),
            "final_novel_algorithm_report.json": ("report", "Novel Algorithm Report"),
        }
        # NoSkel-Agents planner files
        AGENT_MAP_NOSKEL = {
            "noskel_algorithm_spec.json": ("design", "Algorithm Design Spec"),
            "noskel_design_session.json": ("design_session", "Design Session Transcript"),
            "agent8_final_synthesis.json": ("8f", "Code Synthesis"),
            "noskel_metrics.json": ("noskel_metrics", "NoSkel-Agents Algorithm Metrics"),
            "noskel_comparison.json": ("noskel_cmp", "NoSkel-Agents vs Baselines Comparison"),
            "final_noskel_report.json": ("report", "Final Report"),
        }
        # SingleLLM planner files
        AGENT_MAP_SINGLELLM = {
            "singlellm_design.json": ("design", "Blank-Prompt Algorithm Design"),
            "singlellm_comparison.json": ("singlellm_cmp", "SingleLLM vs Baselines Comparison"),
            "final_singlellm_report.json": ("report", "Final Report"),
        }
        # Auto-Skills planner files (per-agent files reuse AGENT_MAP_NOVEL)
        AGENT_MAP_AUTO_SKILLS = {
            "final_auto_skills_report.json": ("report", "Final Report"),
        }
        # Combined map
        AGENT_MAP = {**AGENT_MAP_OPT, **AGENT_MAP_NOVEL, **AGENT_MAP_NOSKEL, **AGENT_MAP_SINGLELLM, **AGENT_MAP_AUTO_SKILLS}
        # Skill tracking files
        SKILL_LOG_PATTERN = r"_skills\.json$"
        if fname in AGENT_MAP:
            agent_id, agent_name = AGENT_MAP[fname]
            try:
                with open(fpath) as f:
                    data_obj = json.load(f)
                if agent_id == "history":
                    events.append({"type": "history", "data": {"history": data_obj}})
                else:
                    events.append({"type": "agent_result", "data": {
                        "agent": agent_id, "name": agent_name,
                        "filename": fname, "result": data_obj,
                    }})
            except Exception:
                pass
            continue

        # --- Phase plots ---
        PHASE_PLOTS = {
            "phase1_data_overview.png": "Data Overview",
            "phase2a_difficulty_map.png": "Difficulty Map",
            "phase2b_outlier_map.png": "Outlier Map",
            "phase4_convergence.png": "Convergence Curve",
            "phase6_distortion_heatmap.png": "Distortion Heatmap",
            "phase6_highlighted_areas.png": "Highlighted Areas",
            "phase6_class_distortion.png": "Class Distortion",
            "phase6_dendrogram_comparison.png": "Dendrogram Comparison",
            "phase6_hierarchy_changes.png": "Hierarchy Changes",
            # Optimize-mode comparison plots
            "default_params_embedding.png": "Default Parameters Embedding",
            "default_vs_best_comparison.png": "Default vs Best Comparison",
            # AutoManifold Phase 6 plots
            "novel_embedding.png": "Novel Algorithm Embedding",
            "novel_comparison.png": "Novel vs Baselines",
            "novel_metrics_table.png": "Metrics Comparison Table",
            "novel_metrics_bars.png": "Metrics Bar Chart",
            "noskel_comparison.png": "NoSkel-Agents vs Baselines",
            "noskel_metrics_table.png": "Metrics Comparison Table",
            "noskel_metrics_bars.png": "Metrics Bar Chart",
            "singlellm_comparison.png": "SingleLLM vs Baselines",
            "singlellm_metrics_table.png": "Metrics Comparison Table",
            "singlellm_metrics_bars.png": "Metrics Bar Chart",
        }
        if fname in PHASE_PLOTS:
            events.append({"type": "phase_plot", "data": {
                "name": fname.replace(".png", ""), "label": PHASE_PLOTS[fname], "url": url,
            }})

        # --- Skill tracking logs ---
        if re.match(SKILL_LOG_PATTERN, fname):
            try:
                with open(fpath) as f:
                    skill_data = json.load(f)
                events.append({"type": "skill_log", "data": {
                    "filename": fname, "url": url, "summary": skill_data
                }})
            except Exception:
                pass

    # --- Per-turn skill artifacts (Auto-Skills) ---
    skill_turns_dir = os.path.join(output_dir, "skill_turns")
    if os.path.isdir(skill_turns_dir):
        for fn in sorted(os.listdir(skill_turns_dir)):
            if not fn.endswith(".json"):
                continue
            seen_key = f"skill_turns/{fn}"
            if seen_key in seen:
                continue
            seen.add(seen_key)
            fpath = os.path.join(skill_turns_dir, fn)
            try:
                with open(fpath) as f:
                    turn_data = json.load(f)
                events.append({"type": "agent_result", "data": {
                    "agent": "skill_turn",
                    "name": f"Skill turn — {fn[:-5]}",
                    "filename": seen_key,
                    "result": turn_data,
                }})
            except Exception:
                pass

    return events


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{run_id}")
async def ws_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()

    if run_id not in runs:
        await websocket.send_json({"type": "error", "data": {"message": "Run not found"}})
        await websocket.close()
        return

    run = runs[run_id]
    q: queue.Queue = run["q"]
    output_dir = run["output_dir"]
    seen_files: set = set()

    await websocket.send_json({"type": "init", "data": {
        "run_id": run_id,
        "config": run["config"],
        "dataset": run.get("dataset_info", {}),
    }})

    last_file_check = 0.0
    done = False

    try:
        while not done:
            # --- Drain all pending queue items first ---
            drained = 0
            while drained < 30:
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                if event.get("type") == "_sentinel":
                    done = True
                    break
                await websocket.send_json(event)

            # --- File watcher (every 600 ms) ---
            now = time.monotonic()
            if now - last_file_check > 0.6:
                last_file_check = now
                for ev in _check_new_files(output_dir, seen_files, run_id):
                    await websocket.send_json(ev)

            if not done:
                await asyncio.sleep(0.08)

        # Flush any remaining file events
        for ev in _check_new_files(output_dir, seen_files, run_id):
            await websocket.send_json(ev)
        await websocket.send_json({"type": "done", "data": {}})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
async def api_datasets():
    from fastapi.concurrency import run_in_threadpool
    def _load():
        from pipeline.datasets import get_dataset_catalog
        return get_dataset_catalog()
    return await run_in_threadpool(_load)


@app.post("/api/start")
async def api_start(
    dataset: Optional[str] = Form(None),
    provider: str = Form("google"),
    model: str = Form("gemini-3.1-pro-preview"),
    api_key: str = Form(""),
    experiment_name: str = Form(""),
    mode: str = Form("optimization"),  # a key of MODES
    dr_method: str = Form("tsne"),
    max_iter: int = Form(10),
    quality_target: float = Form(9.0),
    iterative_optimization: bool = Form(True),
    primary_goal: str = Form("balance both"),
    data_file: Optional[UploadFile] = File(None),
    labels_file: Optional[UploadFile] = File(None),
):
    if mode not in MODES:
        raise HTTPException(400, f"Unknown mode {mode!r}; expected one of {sorted(MODES)}")

    # Resolve API key
    if not api_key:
        api_key = os.environ.get({
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }.get(provider, "OPENAI_API_KEY"), "")
    if not api_key:
        raise HTTPException(400, f"No API key provided for {provider}. Set env var or pass key.")

    # Build run ID + output dir
    run_id = _make_run_id(experiment_name)
    output_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(output_dir, exist_ok=True)

    dataset_info: dict = {}

    # --- Load data ---
    if dataset:
        from pipeline.datasets import get_dataset, get_dataset_availability
        try:
            availability = get_dataset_availability(dataset)
            if not availability.get("available", True):
                raise HTTPException(400, availability.get("reason") or f"Dataset '{dataset}' is unavailable")
            data_arr, labels_arr, ln_list, ds_meta = get_dataset(dataset)
            label_names = {i: name for i, name in enumerate(ln_list)}
            dataset_info = ds_meta
            dataset_info["dataset_key"] = dataset
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, str(e))

    elif data_file and data_file.filename:
        contents = await data_file.read()
        suffix = Path(data_file.filename).suffix.lower()
        tmp_data = os.path.join(UPLOAD_ROOT, f"{run_id}_data{suffix}")
        with open(tmp_data, "wb") as f:
            f.write(contents)

        import pandas as pd
        if suffix == ".npy":
            data_arr = np.load(tmp_data).astype(np.float64)
        else:
            df = pd.read_csv(tmp_data)
            id_cols = [c for c in df.columns if c.lower() in ("id", "sample_id", "index")]
            df = df.drop(columns=id_cols, errors="ignore")
            numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all").dropna()
            data_arr = numeric.values.astype(np.float64)
        dataset_info = {"name": data_file.filename, "n_samples": int(len(data_arr)),
                        "n_features": int(data_arr.shape[1])}

        labels_arr = None
        label_names = None

        if labels_file and labels_file.filename:
            lbl_contents = await labels_file.read()
            lbl_suffix = Path(labels_file.filename).suffix.lower()
            tmp_lbl = os.path.join(UPLOAD_ROOT, f"{run_id}_labels{lbl_suffix}")
            with open(tmp_lbl, "wb") as f:
                f.write(lbl_contents)
            if lbl_suffix == ".npy":
                labels_arr = np.load(tmp_lbl).astype(np.int32)
                label_names = None
            else:
                ldf = pd.read_csv(tmp_lbl)
                id_cols = [c for c in ldf.columns if c.lower() in ("id", "sample_id", "index")]
                ldf = ldf.drop(columns=id_cols, errors="ignore")
                non_num = ldf.select_dtypes(exclude=[np.number]).columns.tolist()
                col = non_num[0] if non_num else ldf.columns[0]
                series = ldf[col].astype(str)
                unique = sorted(series.unique())
                label_names = {i: n for i, n in enumerate(unique)}
                labels_arr = np.array([unique.index(v) for v in series], dtype=np.int32)
    else:
        raise HTTPException(400, "Provide either a built-in dataset key or upload a data file.")

    cfg = {
        "provider": provider, "model": model, "api_key": api_key,
        "experiment_name": experiment_name.strip(),
        "mode": mode,
        "dr_method": dr_method, "max_iter": max_iter,
        "quality_target": quality_target, "primary_goal": primary_goal,
        "iterative_optimization": iterative_optimization,
    }

    q: queue.Queue = queue.Queue()
    run_state = {
        "run_id": run_id,
        "status": "starting",
        "output_dir": output_dir,
        "q": q,
        "config": {k: v for k, v in cfg.items() if k != "api_key"},  # never expose key
        "dataset_info": dataset_info,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    runs[run_id] = run_state

    # Persist config so past-run loader can display it
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if k != "api_key"}, f, indent=2)

    t = threading.Thread(
        target=_run_pipeline,
        args=(run_id, cfg, data_arr, labels_arr, label_names, output_dir),
        daemon=True,
    )
    t.start()

    return {"run_id": run_id}


@app.get("/api/runs")
async def api_runs():
    result = []
    # Live in-memory runs
    for run_id, run in runs.items():
        result.append({
            "run_id": run_id, "status": run["status"],
            "config": run.get("config", {}),
            "dataset": run.get("dataset_info", {}),
            "created_at": run.get("created_at", run_id),
        })
    # Past runs — directory names only, no file reading
    live_ids = set(runs.keys())
    if os.path.isdir(OUTPUT_ROOT):
        for dname in sorted(os.listdir(OUTPUT_ROOT), reverse=True)[:100]:
            if dname in live_ids:
                continue
            if os.path.isdir(os.path.join(OUTPUT_ROOT, dname)):
                result.append({"run_id": dname, "past": True})
    return result


@app.get("/api/run/{run_id}/snapshot")
async def api_run_snapshot(run_id: str):
    """Return all file-based events from a run's output directory (for past-run replay)."""
    from fastapi.concurrency import run_in_threadpool
    output_dir = runs[run_id]["output_dir"] if run_id in runs else os.path.join(OUTPUT_ROOT, run_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(404, "Run not found")
    def _load():
        events = list(_check_new_files(output_dir, set(), run_id))

        # Sort events so they replay in the correct phase/iteration order
        TYPE_ORDER = {
            'agent_result': 1,  # phase agents (further sorted by agent id)
            'phase_plot': 2,    # phase-level plots
            'metrics': 3,       # per-iteration metrics
            'plot': 4,          # per-iteration embedding images
            'agent6': 5,        # per-iteration agent 6
            'agent7': 6,        # per-iteration agent 7
            'history': 7,       # optimization history (needs all iterations first)
        }
        AGENT_ORDER = {
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '4A': 4, '5C': 5,  # Novel planner agents
            '8i': 6, '8f': 8, '9': 9,
            'novel_metrics': 10, 'novel_cmp': 11,  # Novel planner results
            'noskel_metrics': 12, 'noskel_cmp': 13,
        }
        PHASE_PLOT_ORDER = {
            'phase1_data_overview': 1,
            'phase2a_difficulty_map': 2, 'phase2b_outlier_map': 3,
            'phase4_convergence': 4,
            'phase6_distortion_heatmap': 5, 'phase6_highlighted_areas': 6,
            'phase6_class_distortion': 7, 'phase6_dendrogram_comparison': 8,
            'phase6_hierarchy_changes': 9,
            # Novel planner Phase 6
            'novel_embedding': 10, 'novel_comparison': 11,
            'noskel_comparison': 12, 'noskel_metrics_table': 13,
            'noskel_metrics_bars': 14,
        }

        def sort_key(ev):
            t = ev.get("type", "")
            d = ev.get("data", {})
            primary = TYPE_ORDER.get(t, 99)
            if t == 'agent_result':
                return (primary, AGENT_ORDER.get(d.get("agent", ""), 99), 0)
            if t == 'phase_plot':
                return (primary, PHASE_PLOT_ORDER.get(d.get("name", ""), 99), 0)
            if t in ('plot', 'agent6', 'agent7', 'metrics'):
                return (primary, d.get("iteration", 0), 0)
            return (primary, 0, 0)

        events.sort(key=sort_key)

        cfg_data = {}
        cfg_path = os.path.join(output_dir, "run_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg_data = json.load(f)
            except Exception:
                pass
        return {"run_id": run_id, "config": cfg_data, "events": events}
    return await run_in_threadpool(_load)


@app.get("/api/run/{run_id}/history")
async def api_run_history(run_id: str):
    from fastapi.concurrency import run_in_threadpool
    output_dir = runs[run_id]["output_dir"] if run_id in runs else os.path.join(OUTPUT_ROOT, run_id)
    path = os.path.join(output_dir, "optimization_history.json")
    if not os.path.exists(path):
        raise HTTPException(404, "History not found")
    def _load():
        with open(path) as f:
            return json.load(f)
    return await run_in_threadpool(_load)


@app.get("/api/run/{run_id}/skills")
async def api_run_skills(run_id: str):
    """Get skill usage log for a run."""
    from fastapi.concurrency import run_in_threadpool
    output_dir = runs[run_id]["output_dir"] if run_id in runs else os.path.join(OUTPUT_ROOT, run_id)

    # Check for skill log in main output dir
    skill_log_path = os.path.join(output_dir, f"{run_id}_skills.json")
    if os.path.exists(skill_log_path):
        def _load():
            with open(skill_log_path) as f:
                return json.load(f)
        return await run_in_threadpool(_load)

    # Check in skill_logs subdirectory
    skill_log_path = os.path.join(OUTPUT_ROOT, "skill_logs", f"{run_id}_skills.json")
    if os.path.exists(skill_log_path):
        def _load():
            with open(skill_log_path) as f:
                return json.load(f)
        return await run_in_threadpool(_load)

    raise HTTPException(404, "Skill log not found")


@app.get("/files/{run_id}/{filename:path}")
async def serve_file(run_id: str, filename: str):
    output_dir = runs[run_id]["output_dir"] if run_id in runs else os.path.join(OUTPUT_ROOT, run_id)
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)


# Serve frontend
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"Starting DR Design Pipeline server at http://localhost:{args.port}")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, log_level="warning")
