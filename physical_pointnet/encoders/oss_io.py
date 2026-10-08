"""OSS バックボーン向けに PointBatch を固定点数テンソルへ揃える。"""

from typing import Tuple

import torch

from ..points import PointBatch


def pointbatch_to_fixed(
    points: PointBatch, num_point: int
):
    # type: (PointBatch, int) -> Tuple[torch.Tensor, torch.Tensor]
    """サンプルごとに ``num_point`` 点へ揃える。

    返り値は座標 ``(B, num_point, 3)`` と特徴 ``(B, num_point, C)``。
    点が足りないときは先頭から繰り返して埋め、多いときは等間隔で間引く。
    （公開分類実装が固定点数を前提にしているための前処理。）
    """
    device = points.coord.device
    dtype = points.coord.dtype
    feat_dim = points.feat.shape[-1]
    xyz_out = torch.empty(points.batch_size, num_point, 3, device=device, dtype=dtype)
    feat_out = torch.empty(points.batch_size, num_point, feat_dim, device=device, dtype=dtype)
    for b in range(points.batch_size):
        mask = points.batch == b
        xyz_b = points.coord[mask]
        feat_b = points.feat[mask]
        n = int(xyz_b.shape[0])
        if n <= 0:
            raise ValueError("点が 0 個のサンプルがあります。")
        if n == num_point:
            xyz_out[b] = xyz_b
            feat_out[b] = feat_b
        elif n > num_point:
            idx = torch.linspace(0, n - 1, num_point, device=device).long()
            xyz_out[b] = xyz_b[idx]
            feat_out[b] = feat_b[idx]
        else:
            reps = (num_point + n - 1) // n
            xyz_out[b] = xyz_b.repeat(reps, 1)[:num_point]
            feat_out[b] = feat_b.repeat(reps, 1)[:num_point]
    return xyz_out, feat_out
