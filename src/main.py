"""Orchestrator for a single run_id (Hydra entrypoint).

Applies the mode overrides and launches `src/inference.py` as a subprocess.
No scoring, aggregation or evaluation logic lives here (AGENTS.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Only the scale changes between modes; the dataset, the search space, the
# proxies and the aggregation rule are identical everywhere (AGENTS.md).
#
# `full` samples 300 of the 15,625 architectures. That is smaller than the
# 1,000 NAS-Bench-Suite-Zero uses, and the reason is cost: run time scales with
# the architecture count, not with the GPU. At 300 the true top 10% is 30
# architectures, the floor at which the top-10% metrics still mean anything.
MODE_OVERRIDES = {
    # Cheap enough to run locally on CPU.
    "sanity": {"n_archs": 8, "seeds": [0], "batch_size": 16, "search_budget": 5},
    # ~27% of the full architecture budget.
    "pilot": {"n_archs": 80, "seeds": [0, 1, 2], "batch_size": 128, "search_budget": 30},
    "full": {"n_archs": 300, "seeds": [0, 1, 2], "batch_size": 128, "search_budget": 30},
}


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    mode = str(cfg.mode)
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"mode must be one of {sorted(MODE_OVERRIDES)}, got {mode!r}")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    payload = {
        "mode": mode,
        "results_dir": str(cfg.results_dir),
        "cache_dir": str(cfg.cache_dir),
        "shared_cache_dir": str(cfg.shared_cache_dir) if cfg.shared_cache_dir else "",
        "dataset": str(cfg.dataset),
        "num_classes": int(cfg.num_classes),
        "dataload_batches": int(cfg.dataload_batches),
        "sample_seed": int(cfg.sample_seed),
        "primary_seed_index": int(cfg.primary_seed_index),
        "run_id": str(cfg.run.run_id),
        "method": str(cfg.run.method),
        "proxies": list(cfg.run.proxies),
        **MODE_OVERRIDES[mode],
    }

    # sanity and pilot must never pollute the full runs' W&B namespace.
    project = (resolved.get("wandb") or {}).get("project")
    if project and mode in ("sanity", "pilot"):
        project = f"{project}-{mode}"
    payload["wandb_project"] = project
    payload["wandb_entity"] = (resolved.get("wandb") or {}).get("entity")

    print(f"[main] run_id={payload['run_id']} mode={mode} "
          f"method={payload['method']} n_archs={payload['n_archs']} "
          f"seeds={payload['seeds']}", flush=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        config_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-m", "src.inference", config_path],
            cwd=str(Path(__file__).resolve().parent.parent),
            check=False,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
