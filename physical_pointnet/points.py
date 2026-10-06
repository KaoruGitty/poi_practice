"""点数がサンプルごとに違う点群のバッチと、エンコーダが共有する点群演算。

演算はすべて PyTorch だけで書いてあり、CPU でも CUDA でも同じコードで動く。
大きな点群で速さが足りなくなったら、このファイルの関数を torch_cluster や pyg-lib の
CUDA 実装に差し替える。
"""

from __future__ import annotations

from typing import Union, List, Tuple

from dataclasses import dataclass

import numpy as np
import torch

@dataclass
class PointBatch:
    """全サンプルの点を縦につないだバッチ。

    batch[i] は点 i が属するサンプル番号で、昇順に並んでいる必要がある。
    各サンプルには少なくとも 1 点ある。
    """

    coord: torch.Tensor  # (M, 3) ロボット座標のメートル
    feat: torch.Tensor  # (M, C) 点ごとの入力特徴
    batch: torch.Tensor  # (M,) long
    batch_size: int

    @property
    def counts(self) -> torch.Tensor:
        return torch.bincount(self.batch, minlength=self.batch_size)

    def to(self, device: Union[torch.device, str]) -> "PointBatch":
        return PointBatch(
            self.coord.to(device, non_blocking=True),
            self.feat.to(device, non_blocking=True),
            self.batch.to(device, non_blocking=True),
            self.batch_size,
        )

    @staticmethod
    def from_arrays(coords: List[np.ndarray], feats: List[np.ndarray]) -> "PointBatch":
        counts = [len(c) for c in coords]
        if min(counts) < 1:
            raise ValueError("点が 0 個のサンプルがあります。前処理で最低 1 点を残してください。")
        batch = np.repeat(np.arange(len(counts)), counts)
        return PointBatch(
            coord=torch.from_numpy(np.concatenate(coords).astype(np.float32, copy=False)),
            feat=torch.from_numpy(np.concatenate(feats).astype(np.float32, copy=False)),
            batch=torch.from_numpy(batch.astype(np.int64)),
            batch_size=len(counts),
        )

def to_padded(
    values: torch.Tensor, batch: torch.Tensor, batch_size: int, fill: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(M, C) を (B, Nmax, C) に詰め直す。mask は実在する点で True。"""
    counts = torch.bincount(batch, minlength=batch_size)
    start = torch.cumsum(counts, 0) - counts
    local = torch.arange(batch.numel(), device=batch.device) - start[batch]
    nmax = int(counts.max())
    out = values.new_full((batch_size, nmax, values.shape[-1]), fill)
    out[batch, local] = values
    mask = torch.zeros(batch_size, nmax, dtype=torch.bool, device=batch.device)
    mask[batch, local] = True
    return out, mask

def gather_points(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """values (B, N, C) から idx (B, ...) の点を取り出して (B, ..., C) にする。"""
    batch = torch.arange(values.shape[0], device=values.device)
    batch = batch.view(-1, *([1] * (idx.dim() - 1))).expand_as(idx)
    return values[batch, idx]

def farthest_point_sample(
    xyz: torch.Tensor, mask: torch.Tensor, npoint: int, random_start: bool = False
) -> torch.Tensor:
    """詰めた点群 (B, N, 3) から npoint 点を最遠点サンプリングで選ぶ。

    実在する点が npoint より少ないサンプルでは、先頭の実在点数ぶんだけが別々の点で、
    残りは重複する。呼び出し側で先頭だけを有効として扱う。
    """
    batch, count, _ = xyz.shape
    rows = torch.arange(batch, device=xyz.device)
    distance = torch.where(mask, torch.full_like(mask, float("inf"), dtype=xyz.dtype), -1.0)
    if random_start:
        farthest = torch.multinomial(mask.to(xyz.dtype), 1).squeeze(1)
    else:
        farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    chosen = torch.empty(batch, npoint, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        chosen[:, i] = farthest
        d = ((xyz - xyz[rows, farthest].unsqueeze(1)) ** 2).sum(-1)
        distance = torch.where(mask, torch.minimum(distance, d), -1.0)
        farthest = distance.argmax(dim=1)
    return chosen

def ball_query(
    centers: torch.Tensor,
    center_mask: torch.Tensor,
    xyz: torch.Tensor,
    mask: torch.Tensor,
    radius: float,
    k: int,
    chunk_elems: int = 1 << 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """各中心から近い順に最大 k 点を取り、半径の外と存在しない点を無効にする。

    距離表は中心をいくつかずつに区切って作るので、メモリは chunk_elems 要素程度に収まる。
    返り値は近傍の番号 (B, S, k) と有効フラグ (B, S, k)。
    """
    batch, nsrc, _ = centers.shape
    count = xyz.shape[1]
    k_eff = min(k, count)
    idx = torch.zeros(batch, nsrc, k, dtype=torch.long, device=xyz.device)
    valid = torch.zeros(batch, nsrc, k, dtype=torch.bool, device=xyz.device)
    step = max(1, chunk_elems // max(1, batch * count))
    invalid = ~mask.unsqueeze(1)
    for s0 in range(0, nsrc, step):
        s1 = min(nsrc, s0 + step)
        d = torch.cdist(centers[:, s0:s1], xyz).masked_fill(invalid, float("inf"))
        dist, nn_idx = d.topk(k_eff, dim=-1, largest=False)
        idx[:, s0:s1, :k_eff] = nn_idx
        valid[:, s0:s1, :k_eff] = dist <= radius
    valid &= center_mask.unsqueeze(-1)
    return idx, valid

def segment_reduce(values: torch.Tensor, index: torch.Tensor, size: int, reduce: str) -> torch.Tensor:
    """index ごとに values (M, C) を max / min / mean でまとめて (size, C) にする。"""
    if reduce == "mean":
        out = values.new_zeros(size, values.shape[-1]).index_add_(0, index, values)
        count = torch.bincount(index, minlength=size).clamp(min=1).to(values.dtype)
        return out / count.unsqueeze(-1)
    fill = float("-inf") if reduce == "max" else float("inf")
    out = values.new_full((size, values.shape[-1]), fill)
    mode = "amax" if reduce == "max" else "amin"
    return out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(values), values, mode, include_self=True)

def global_pool(feat: torch.Tensor, batch: torch.Tensor, batch_size: int, mode: str) -> torch.Tensor:
    """点ごとの特徴をサンプルごとに 1 本のベクトルにする。mode は max、mean、max+mean。"""
    if mode == "max+mean":
        return torch.cat(
            [segment_reduce(feat, batch, batch_size, "max"), segment_reduce(feat, batch, batch_size, "mean")],
            dim=1,
        )
    return segment_reduce(feat, batch, batch_size, mode)

def pooled_width(channels: int, mode: str) -> int:
    return channels * 2 if mode == "max+mean" else channels
