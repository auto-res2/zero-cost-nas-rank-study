"""In-platform scoring: run airas-eval over every restored eval_inputs set.

Runs on the execution platform so that evaluation/*.json and metrics.json are
platform-stored run outputs, which is what the paper-values provenance check
requires. No metric is computed here: scoring and aggregation are the pinned
airas-eval CLI (invoked as a subprocess, never imported), and metrics.json is
a pure transcription of its outputs.

Invoked as `python .research/score_all.py` (plain argv — no shell, so the
image's PATH containing /workspace/.venv/bin applies and no login profile can
reset it; an earlier bash -lc attempt failed on exactly that).
"""
import json
import pathlib
import subprocess
import sys

RESULTS = pathlib.Path(".research/results")


def run(argv):
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    run_dirs = sorted(
        d.parent.parent for d in RESULTS.glob("*/eval_inputs/nas_pre_training.json")
    )
    if not run_dirs:
        print("no eval_inputs found under", RESULTS, flush=True)
        return 1

    for base in run_dirs:
        (base / "evaluation").mkdir(parents=True, exist_ok=True)
        for f in sorted((base / "eval_inputs").glob("nas_pre_training*.json")):
            out = base / "evaluation" / f.name
            run(["airas-eval", "score", "nas_pre_training",
                 "--inputs", str(f), "--output", str(out)])
        run(["airas-eval", "aggregate", "--label", base.name,
             "--reports",
             str(base / "evaluation/nas_pre_training.seed0.json"),
             str(base / "evaluation/nas_pre_training.seed1.json"),
             str(base / "evaluation/nas_pre_training.seed2.json"),
             "--output", str(base / "evaluation/aggregate.json")])

        # Transcription (copying only): assemble metrics.json from the
        # airas-eval outputs above plus the run-produced raw inputs.
        m = {}
        for s in (0, 1, 2):
            m[f"seed{s}"] = json.load(open(base / f"evaluation/nas_pre_training.seed{s}.json"))["metrics"]
        m["canonical"] = json.load(open(base / "evaluation/nas_pre_training.json"))["metrics"]
        m["agg"] = json.load(open(base / "evaluation/aggregate.json"))["metrics"]
        inp = json.load(open(base / "eval_inputs/nas_pre_training.json"))
        m["inputs"] = {"oracle_best": inp["oracle_best"],
                       "n_candidates": len(inp["predicted_scores"]),
                       "n_evaluations": len(inp["evaluated_scores"])}
        json.dump(m, open(base / "metrics.json", "w"), indent=1)
        print(base.name, "-> metrics.json", flush=True)

    print(f"scored {len(run_dirs)} run directories", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
