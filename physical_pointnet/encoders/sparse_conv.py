"""ボクセル上の部分多様体の疎な 3D 畳み込み（submanifold sparse convolution）。

出力は入力と同じボクセルにだけ作る。近傍のボクセルは番号表で引き、存在しないものは 0 として扱う。
spconv の SubMConv3d と同じ計算を PyTorch だけで行う。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def neighbor_table(grid: torch.Tensor, batch: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """各ボクセルについて、kernel_size^3 個の近傍ボクセルの番号を返す。無ければ M。

    grid はサンプルの中で重複しない非負整数座標 (M, 3)。
    """
    r = kernel_size // 2
    steps = torch.arange(-r, r + 1, device=grid.device)
    offsets = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), dim=-1).reshape(-1, 3)
    shifted = grid + r
    size = int(shifted.max()) + r + 1

    def key(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((b * size + g[..., 0]) * size + g[..., 1]) * size + g[..., 2]

    count = grid.shape[0]
    sorted_keys, perm = key(shifted, batch).sort()
    query = key(shifted.unsqueeze(1) + offsets.unsqueeze(0), batch.unsqueeze(1))
    pos = torch.searchsorted(sorted_keys, query.reshape(-1)).clamp(max=count - 1).view_as(query)
    found = sorted_keys[pos] == query
    return torch.where(found, perm[pos], torch.full_like(pos, count))


class SubMConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size は奇数にしてください。")
        self.kernel_size = kernel_size
        volume = kernel_size**3
        self.weight = nn.Parameter(torch.empty(volume * in_channels, out_channels))
        bound = 1.0 / math.sqrt(volume * in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, feat: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([feat, feat.new_zeros(1, feat.shape[1])], dim=0)
        out = padded[table].reshape(feat.shape[0], -1) @ self.weight.to(feat.dtype)
        if self.bias is not None:
            out = out + self.bias.to(feat.dtype)
        return out
