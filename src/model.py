"""NAS-Bench-201 cell network, rebuilt from the published search-space spec.

REUSE vs OWN IMPLEMENTATION
---------------------------
The macro skeleton, the operation set and the cell wiring below are a
faithful re-implementation of the NAS-Bench-201 search space as defined in

    Dong & Yang, "NAS-Bench-201: Extending the Scope of Reproducible Neural
    Architecture Search", ICLR 2020 (arXiv:2001.00326)
    reference code: https://github.com/D-X-Y/NAS-Bench-201
                    (xautodl/models/cell_operations.py,
                     xautodl/models/cell_infers/{cells,tiny_network}.py)

It is re-implemented rather than imported because the official package
(`xautodl`) pulls a large dependency tree and is not needed here: we only ever
build a randomly initialized network and run a single forward/backward pass —
no training, no checkpoint loading. Everything in this file is therefore
"published spec, re-implemented"; nothing here is a contribution of this study.
The novel part lives only in the aggregation function in `src/inference.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The five candidate operations of the NAS-Bench-201 cell (Dong & Yang 2020,
# Sec. 3.1). Order is fixed by the benchmark and must not be changed.
OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")

# Cell topology: 4 nodes, node i takes one edge from every node j < i.
N_NODES = 4


class ReLUConvBN(nn.Module):
    """ReLU -> Conv -> BN, the `nor_conv_*` operation of NAS-Bench-201."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int = 1) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Zero(nn.Module):
    """The `none` operation: drops the edge by emitting zeros."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(0.0)


class Identity(nn.Module):
    """The `skip_connect` operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_op(name: str, channels: int) -> nn.Module:
    if name == "none":
        return Zero()
    if name == "skip_connect":
        return Identity()
    if name == "nor_conv_1x1":
        return ReLUConvBN(channels, channels, 1)
    if name == "nor_conv_3x3":
        return ReLUConvBN(channels, channels, 3)
    if name == "avg_pool_3x3":
        # count_include_pad=False matches the official implementation.
        return nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
    raise ValueError(f"unknown NAS-Bench-201 operation: {name!r}")


def parse_arch(arch_str: str) -> list[list[tuple[str, int]]]:
    """Parse the canonical NAS-Bench-201 architecture string.

    Format (the exact keys used by the benchmark table), e.g.

        |nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|...|

    Groups separated by '+' describe nodes 1, 2, 3; inside a group each
    `op~j` is the operation on the edge from node j.
    """
    nodes: list[list[tuple[str, int]]] = []
    for group in arch_str.split("+"):
        edges = [e for e in group.strip().strip("|").split("|") if e]
        parsed = []
        for edge in edges:
            op_name, _, src = edge.partition("~")
            if op_name not in OPS:
                raise ValueError(f"unknown operation {op_name!r} in {arch_str!r}")
            parsed.append((op_name, int(src)))
        nodes.append(parsed)
    if len(nodes) != N_NODES - 1:
        raise ValueError(f"expected {N_NODES - 1} node groups, got {len(nodes)}")
    return nodes


class InferCell(nn.Module):
    """One NAS-Bench-201 cell: a DAG over 4 nodes with a fixed channel count."""

    def __init__(self, arch: list[list[tuple[str, int]]], channels: int) -> None:
        super().__init__()
        self.edges = nn.ModuleList()
        self.layout: list[list[tuple[int, int]]] = []
        for node_edges in arch:
            entry = []
            for op_name, src in node_edges:
                entry.append((len(self.edges), src))
                self.edges.append(build_op(op_name, channels))
            self.layout.append(entry)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for entry in self.layout:
            states.append(sum(self.edges[idx](states[src]) for idx, src in entry))
        return states[-1]


class ResNetBasicblock(nn.Module):
    """Stride-2 residual block used between stages (Dong & Yang 2020)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv_a = ReLUConvBN(c_in, c_out, 3, stride=2)
        self.conv_b = ReLUConvBN(c_out, c_out, 3)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0),
            nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x) + self.conv_b(self.conv_a(x))


class TinyNetwork(nn.Module):
    """The NAS-Bench-201 macro skeleton: stem, 3 stages of N cells, head."""

    def __init__(self, arch_str: str, channels: int = 16, num_cells: int = 5,
                 num_classes: int = 10) -> None:
        super().__init__()
        arch = parse_arch(arch_str)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        layers: list[nn.Module] = []
        c_cur = channels
        for stage in range(3):
            if stage > 0:
                layers.append(ResNetBasicblock(c_cur, c_cur * 2))
                c_cur *= 2
            for _ in range(num_cells):
                layers.append(InferCell(arch, c_cur))
        self.cells = nn.Sequential(*layers)
        self.lastact = nn.Sequential(nn.BatchNorm2d(c_cur), nn.ReLU(inplace=False))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_cur, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cells(self.stem(x))
        out = self.lastact(out)
        out = self.global_pooling(out).flatten(1)
        return self.classifier(out)
