"""Score architectures with the reference zero-cost proxies, then aggregate.

No architecture is trained. Each one is built at random initialization, scored
with a single forward/backward pass, and discarded.

WHAT IS REUSED vs WHAT IS OURS
-------------------------------
REUSED, unmodified, from the reference implementation of the paper this study
reproduces — Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS",
ICLR 2021 (arXiv:2101.08134), https://github.com/mohsaied/zero-cost-nas
(Apache-2.0, pinned in the Dockerfile to commit b5059bc4):

  foresight.models.nasbench2.get_model_from_arch_str  the NAS-Bench-201 network
  foresight.weight_initializers.init_net              their initialisation
  foresight.pruners.predictive.find_measures          synflow / jacob_cov /
                                                      snip / grad_norm
  foresight.dataset.get_cifar_dataloaders             (via src/preprocess.py)

Nothing about the proxies is re-derived here. That is deliberate: the published
Spearman values (synflow 0.74, jacob_cov 0.73, snip 0.58, grad_norm 0.58 on
NAS-Bench-201 / CIFAR-10) are only a meaningful reproduction target if the
scores come from the authors' own code. Any difference then belongs to the
aggregation function, which is what this study is about.

OURS:
  binarized_vote  Abdelfattah et al.'s vote is described in the paper (Sec.
                  4.1.2) but is not part of the released package, so it is
                  implemented here from that description.
  uniform_borda   the standard Borda count, used as the equal-weight ablation.
  rd_weights /
  rd_wborda       the proposed method. This is the only novel contribution.
  build_eval_inputs / write_eval_inputs
                  the evaluation-input contract.

Reported metrics are NOT computed here. This module writes raw evaluation
inputs; airas-eval scores them. `scipy.stats` appears only inside the proposed
method's own weighting rule, which is part of the method, not the reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata, spearmanr

# The four proxies, named exactly as the reference package names them.
PROXY_NAMES = ("synflow", "jacob_cov", "snip", "grad_norm")


def score_architecture(arch_str: str, proxies: tuple[str, ...], train_loader,
                       device: torch.device, num_classes: int,
                       dataload_batches: int) -> dict[str, float]:
    """Compute the requested proxies for one architecture — all reference code.

    Mirrors the reference driver `nasbench2_pred.py`: build the network from the
    architecture string, apply their initialisation, then hand it to
    `find_measures` with one random minibatch.
    """
    from foresight.models import nasbench2
    from foresight.pruners import predictive
    from foresight.weight_initializers import init_net

    net = nasbench2.get_model_from_arch_str(arch_str, num_classes)
    net.to(device)
    # 'none' keeps PyTorch's default initialisation, as in nasbench2_pred.py.
    init_net(net, "none", "none")

    measures = predictive.find_measures(
        net,
        train_loader,
        ("random", dataload_batches, num_classes),
        device,
        measure_names=list(proxies),
    )
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {name: float(measures[name]) for name in proxies}


# --------------------------------------------------------------------------
# Aggregation. Every function consumes the SAME score tensor and returns a
# score per architecture where higher is better.
# --------------------------------------------------------------------------
def _desc_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = best. Ties get their average rank."""
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    return rankdata(-finite, method="average")


def single_proxy(tensor: np.ndarray, proxy_index: int, seed_index: int) -> np.ndarray:
    """Aggregation-free floor: the proxy's own ordering."""
    scores = tensor[proxy_index, seed_index]
    return np.where(np.isfinite(scores), scores, -np.inf)


def binarized_vote(tensor: np.ndarray, seed_index: int) -> np.ndarray:
    """Abdelfattah et al. (2021) majority vote, from the paper's description.

    Each proxy casts one vote per ordered pair for the higher-scoring
    architecture; exact ties cast no vote. The total vote count of architecture
    i equals the number of architectures strictly below it, summed over
    proxies, which ascending min-ranks give directly.
    """
    votes = np.zeros(tensor.shape[2], dtype=np.float64)
    for proxy in range(tensor.shape[0]):
        scores = tensor[proxy, seed_index]
        finite = np.where(np.isfinite(scores), scores, -np.inf)
        votes += rankdata(finite, method="min") - 1.0
    return votes


def uniform_borda(tensor: np.ndarray, seed_index: int) -> np.ndarray:
    """Equal-weight rank average (standard Borda count)."""
    ranks = np.stack([_desc_ranks(tensor[p, seed_index]) for p in range(tensor.shape[0])])
    return -ranks.mean(axis=0)


def rd_weights(tensor: np.ndarray) -> np.ndarray:
    """PROPOSED (novel): label-free reliability x diversity weights.

    reliability s_p  mean Kendall tau of proxy p's rankings across seed pairs
    redundancy  r_p  mean Spearman between p's ranks and the leave-one-out
                     rank average of the other proxies, averaged over seeds
    weight      w_p  proportional to max(s_p, 0) * max(1 - r_p, 0), L1-normalized

    Uses only the proxy score tensor: no ground-truth accuracy, no training,
    no extra forward/backward pass, no tuned hyperparameter.
    """
    n_proxies, n_seeds, _ = tensor.shape
    ranks = np.stack(
        [[_desc_ranks(tensor[p, s]) for s in range(n_seeds)] for p in range(n_proxies)]
    )

    reliability = np.zeros(n_proxies)
    for p in range(n_proxies):
        pairs = [
            kendalltau(ranks[p, i], ranks[p, j]).statistic
            for i in range(n_seeds)
            for j in range(i + 1, n_seeds)
        ]
        pairs = [v for v in pairs if np.isfinite(v)]
        reliability[p] = float(np.mean(pairs)) if pairs else 0.0

    redundancy = np.zeros(n_proxies)
    for p in range(n_proxies):
        others = [q for q in range(n_proxies) if q != p]
        if not others:
            continue
        per_seed = []
        for s in range(n_seeds):
            consensus = ranks[others, s].mean(axis=0)
            value = spearmanr(ranks[p, s], consensus).statistic
            if np.isfinite(value):
                per_seed.append(value)
        redundancy[p] = float(np.mean(per_seed)) if per_seed else 0.0

    weights = np.clip(reliability, 0.0, None) * np.clip(1.0 - redundancy, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return np.full(n_proxies, 1.0 / n_proxies)
    return weights / total


def rd_wborda(tensor: np.ndarray, seed_index: int, weights: np.ndarray) -> np.ndarray:
    """PROPOSED (novel): ranking by the weighted rank sum."""
    ranks = np.stack([_desc_ranks(tensor[p, seed_index]) for p in range(tensor.shape[0])])
    return -(weights @ ranks)


# --------------------------------------------------------------------------
# Evaluation inputs. Raw values only — airas-eval computes every metric.
# --------------------------------------------------------------------------
def build_eval_inputs(predicted: np.ndarray, true_accuracy: list[float],
                      costs: list[float], oracle_best: float,
                      search_budget: int) -> dict:
    """Fill BOTH branches of the nas_pre_training contract.

    predictor branch  predicted_scores + reference_scores
    search branch     evaluated_scores + evaluation_costs + search_space_scores
                      + oracle_best

    The search trajectory is what a training-free NAS run would actually do:
    take the proxy's ranking and look up the true accuracy of the top
    `search_budget` candidates, in that order. It costs nothing extra because
    every value is a table lookup.
    """
    order = np.argsort(-predicted, kind="stable")[:search_budget]
    return {
        "predicted_scores": [float(v) for v in predicted],
        "reference_scores": [float(v) for v in true_accuracy],
        "evaluated_scores": [float(true_accuracy[i]) for i in order],
        "evaluation_costs": [float(costs[i]) for i in order],
        "search_space_scores": [float(v) for v in true_accuracy],
        "oracle_best": float(oracle_best),
    }


def write_eval_inputs(payload: dict, results_dir: str, run_id: str,
                      filename: str = "nas_pre_training.json") -> Path:
    directory = Path(results_dir) / run_id / "eval_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Entry point. `src/main.py` launches this module as a subprocess with the
# path of a resolved-config JSON file (AGENTS.md: main.py orchestrates only).
# --------------------------------------------------------------------------
def _method_scores(method: str, tensor: np.ndarray, seed_index: int,
                   proxies: tuple[str, ...], weights: np.ndarray | None) -> np.ndarray:
    if method == "vote":
        return binarized_vote(tensor, seed_index)
    if method == "borda":
        return uniform_borda(tensor, seed_index)
    if method == "rd_wborda":
        assert weights is not None
        return rd_wborda(tensor, seed_index, weights)
    if method.startswith("single:"):
        return single_proxy(tensor, proxies.index(method.split(":", 1)[1]), seed_index)
    raise ValueError(f"unknown aggregation method: {method!r}")


def run_experiment(cfg: dict) -> int:
    import time

    from src import preprocess

    mode = cfg["mode"]
    run_id = cfg["run_id"]
    results_dir = cfg["results_dir"]
    proxies = tuple(cfg["proxies"])
    seeds = list(cfg["seeds"])
    dataset = cfg["dataset"]
    num_classes = int(cfg["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = preprocess.load_benchmark_table(cfg["cache_dir"])
    oracle_best = preprocess.oracle_best_accuracy(table, dataset)
    archs = preprocess.sample_architectures(table, cfg["n_archs"], cfg["sample_seed"])
    true_accuracy = preprocess.true_accuracies(table, archs, dataset)
    costs = preprocess.training_costs(table, archs, dataset)

    # The reference implementation's loader; built once, re-drawn per architecture.
    train_loader = preprocess.build_train_loader(
        cfg["batch_size"], dataset, cfg["cache_dir"]
    )

    # W&B is best-effort: a logging outage must not lose a finished experiment.
    run = None
    if cfg.get("wandb_project"):
        try:
            import wandb

            run = wandb.init(
                entity=cfg.get("wandb_entity"),
                project=cfg["wandb_project"],
                name=run_id,
                config=cfg,
            )
            print(f"W&B run URL: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001 - logging must never be fatal
            print(f"[warn] W&B disabled ({type(exc).__name__}: {exc})", flush=True)
            run = None

    tensor = np.full((len(proxies), len(seeds), len(archs)), np.nan, dtype=np.float64)
    started = time.time()
    for seed_index, seed in enumerate(seeds):
        for arch_index, arch in enumerate(archs):
            # Seeding here fixes both the weight initialisation and which
            # minibatch find_measures draws, so a seed is reproducible.
            torch.manual_seed(seed * 100_000 + arch_index)
            scores = score_architecture(
                arch, proxies, train_loader, device, num_classes,
                int(cfg["dataload_batches"]),
            )
            for proxy_index, proxy in enumerate(proxies):
                tensor[proxy_index, seed_index, arch_index] = scores[proxy]
            if arch_index and arch_index % 25 == 0:
                done = seed_index * len(archs) + arch_index
                rate = (time.time() - started) / done
                print(f"[progress] seed={seed} scored={arch_index}/{len(archs)} "
                      f"{rate:.3f}s/arch", flush=True)
                if run is not None:
                    try:
                        run.log({"seed": seed, "architectures_scored": arch_index,
                                 "seconds_per_architecture": rate})
                    except Exception:  # noqa: BLE001 - logging must never be fatal
                        pass
    proxy_seconds = time.time() - started

    weights = rd_weights(tensor) if cfg["method"] == "rd_wborda" else None
    if weights is not None:
        print("RD-WBorda weights: "
              + ", ".join(f"{p}={w:.4f}" for p, w in zip(proxies, weights)), flush=True)

    budget = min(cfg["search_budget"], len(archs))
    primary = cfg["primary_seed_index"]
    per_seed_predicted = [
        _method_scores(cfg["method"], tensor, s, proxies, weights)
        for s in range(len(seeds))
    ]

    payload = build_eval_inputs(
        per_seed_predicted[primary], true_accuracy, costs, oracle_best, budget
    )
    official = write_eval_inputs(payload, results_dir, run_id)
    print(f"wrote evaluation inputs: {official}", flush=True)

    # Per-seed inputs for the record. The canonical file above (primary seed,
    # fixed in the config) is the one `make evaluate` scores; these let the
    # seed spread be recomputed by the same evaluator, never by this code.
    for seed_index, seed in enumerate(seeds):
        extra = build_eval_inputs(
            per_seed_predicted[seed_index], true_accuracy, costs, oracle_best, budget
        )
        write_eval_inputs(extra, results_dir, run_id, f"nas_pre_training.seed{seed}.json")

    finite = np.isfinite(per_seed_predicted[primary])
    n_unique = len(np.unique(per_seed_predicted[primary][finite]))
    summary = {
        "steps": int(len(archs)),
        "n_samples": int(len(archs)),
        "unique_scores": int(n_unique),
        "all_finite": bool(finite.all()),
        "proxy_seconds": round(proxy_seconds, 2),
        "seconds_per_architecture": round(
            proxy_seconds / max(len(archs) * len(seeds), 1), 4
        ),
        "device": device.type,
    }
    if weights is not None:
        summary["weights"] = {p: round(float(w), 4) for p, w in zip(proxies, weights)}

    if run is not None:
        try:
            run.summary.update(summary)
            run.summary["oracle_best"] = oracle_best
            run.finish()
        except Exception as exc:  # noqa: BLE001 - logging must never be fatal
            print(f"[warn] W&B summary not written ({exc})", flush=True)

    ok = bool(finite.all()) and n_unique >= 5 and len(archs) >= 5
    if mode == "sanity":
        if ok:
            print("SANITY_VALIDATION: PASS", flush=True)
            print("SANITY_VALIDATION_SUMMARY: " + json.dumps(summary), flush=True)
        else:
            reason = "missing_metrics" if not finite.all() else "identical_outputs"
            print(f"SANITY_VALIDATION: FAIL reason={reason}", flush=True)
            return 1
    elif mode == "pilot":
        # The primary metric is produced by airas-eval, never here, so the pilot
        # gate checks what this code is responsible for: enough scored
        # architectures, all finite, and well-formed evaluation inputs.
        if ok and len(archs) >= 50 and json.loads(official.read_text()):
            print("PILOT_VALIDATION: PASS", flush=True)
            print("PILOT_VALIDATION_SUMMARY: " + json.dumps(summary), flush=True)
        else:
            reason = "insufficient_samples" if len(archs) < 50 else "missing_metrics"
            print(f"PILOT_VALIDATION: FAIL reason={reason}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(run_experiment(json.loads(Path(sys.argv[1]).read_text())))
