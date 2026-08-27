"""Model definition — intentionally empty.

The NAS-Bench-201 network is NOT defined here. It comes from the reference
implementation of the paper this study reproduces:

    foresight.models.nasbench2.get_model_from_arch_str(arch_str, num_classes)

    Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS", ICLR 2021
    https://github.com/mohsaied/zero-cost-nas  (Apache-2.0)

Using the authors' own network builder — rather than re-deriving the search
space from the NAS-Bench-201 paper — is what makes the published Spearman
values a meaningful reproduction target: a difference in the measured
correlation can then only come from the aggregation function under study, not
from a subtly different cell or macro skeleton.

The official NAS-Bench-201 package (`xautodl`) is not used because it pins
numpy<=1.19.5, which cannot coexist with the torch 2.13 build required for
aarch64.
"""
