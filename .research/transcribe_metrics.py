# In-run transcription: assemble metrics.json from airas-eval outputs.
# Pure copying — no metric is computed here. Kept byte-stable (key order,
# indent=1) so the platform-stored bytes match what the paper pipeline reads.
import json, pathlib
for base in sorted(pathlib.Path(".research/results").iterdir()):
    if not (base / "eval_inputs" / "nas_pre_training.json").is_file():
        continue
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
    print(base.name, "-> metrics.json")
