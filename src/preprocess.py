"""Benchmark ground truth and the scoring dataloader.

Two inputs are prepared here and nothing else:

1. The NAS-Bench-201 ground-truth table, read ONLY to score rankings and to
   fill the benchmark-published fields of the evaluation inputs (true accuracy,
   training cost, search-space scores). No architecture is ever trained.

2. The CIFAR-10 training dataloader that the data-dependent proxies consume.
   This is the reference implementation's own loader, not a re-derivation:

       foresight.dataset.get_cifar_dataloaders(...)
       https://github.com/mohsaied/zero-cost-nas  (Apache-2.0)

   It matters that this is theirs: the transforms it applies (RandomCrop with
   padding 4, horizontal flip, and their specific normalisation constants) are
   part of what produced the published Spearman values being reproduced.

REUSE: the ground-truth table is the redistributed NAS-Bench-201 result dump
shipped with NAS-Bench-Suite-Zero (Krishnakumar et al., NeurIPS 2022 D&B),
https://github.com/automl/naslib/tree/zerocost -> naslib/data/nb201_all.pickle
(9.5 MB). The official release is a ~4.7 GB Google-Drive archive with no stable
programmatic URL; this redistribution carries the same 15,625 architectures.
Pinned by SHA-256 so a changed table fails the run instead of the results.
"""

from __future__ import annotations

import hashlib
import pickle
import random
import urllib.request
from pathlib import Path

NB201_URL = (
    "https://raw.githubusercontent.com/automl/naslib/zerocost/naslib/data/nb201_all.pickle"
)
# Pinned 2026-08-27. Verified before every use.
NB201_SHA256 = "b93135dacdf16327733f0c165ee9f431f624e1f896b56b4abdc24df99e5be45e"
NB201_N_ARCHS = 15625


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_table(cache_dir: str) -> dict:
    """Download (once) and return the NAS-Bench-201 table, SHA-256 verified."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "nb201_all.pickle"

    if path.exists() and _sha256(path) != NB201_SHA256:
        path.unlink()
    if not path.exists():
        with urllib.request.urlopen(NB201_URL, timeout=300) as response:
            path.write_bytes(response.read())

    actual = _sha256(path)
    if actual != NB201_SHA256:
        raise RuntimeError(
            f"NAS-Bench-201 table SHA-256 mismatch: expected {NB201_SHA256}, got {actual}"
        )

    with path.open("rb") as handle:
        table = pickle.load(handle)
    if len(table) != NB201_N_ARCHS:
        raise RuntimeError(f"expected {NB201_N_ARCHS} architectures, got {len(table)}")
    return table


def oracle_best_accuracy(table: dict, dataset: str) -> float:
    """Benchmark-published optimum over the WHOLE 15,625-architecture space.

    Fixed by the experimental design and computed from the full table, never
    from the sampled subset and never after looking at a method's ranking.
    """
    return max(float(table[a][dataset]["eval_acc1es"]) for a in table)


def sample_architectures(table: dict, n_archs: int, seed: int) -> list[str]:
    """Uniformly sample architecture strings with a fixed, design-time seed."""
    arch_strings = sorted(table)  # sorted => sampling does not depend on dict order
    if n_archs > len(arch_strings):
        raise ValueError(f"cannot sample {n_archs} of {len(arch_strings)} architectures")
    return random.Random(seed).sample(arch_strings, n_archs)


def true_accuracies(table: dict, archs: list[str], dataset: str) -> list[float]:
    """Published final test accuracy (200 epochs) for each architecture."""
    return [float(table[a][dataset]["eval_acc1es"]) for a in archs]


def training_costs(table: dict, archs: list[str], dataset: str) -> list[float]:
    """Benchmark-published training cost in seconds: per-epoch time x epochs."""
    return [
        float(table[a][dataset]["train_times"]) * float(table[a][dataset]["epochs"])
        for a in archs
    ]


def build_train_loader(batch_size: int, dataset: str, cache_dir: str):
    """The reference implementation's CIFAR-10 training loader, unmodified."""
    from foresight.dataset import get_cifar_dataloaders

    train_loader, _ = get_cifar_dataloaders(
        batch_size, batch_size, dataset, num_workers=0, datadir=cache_dir
    )
    return train_loader
