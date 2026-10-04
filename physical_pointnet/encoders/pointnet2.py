"""PointNet++ エンコーダ。複数半径のセット抽象化（MSG）を重ね、全体を max でまとめる。

半径はメートルで指定する。点数はサンプルごとに違ってよい。
"""

from __future__ import annotations

import torch
from torch import nn

from ..points import PointBatch, ball_query, farthest_point_sample, gather_points, to_padded
from .base import PointEncoder

# このリポジトリのシミュレータ（1 スキャン 300 点弱、最大 8 m）に合わせた既定値。
DEFAULT_SA_LAYERS = (
    {"npoint": 64, "radii": [0.8, 1.6], "nsamples": [8, 16], "mlps": [[32, 32, 64], [64, 64, 128]]},
    {"npoint": 16, "radii": [2.0, 4.0], "nsamples": [8, 16], "mlps": [[64, 64, 128], [128, 128, 256]]},
)


class _SharedMLP(nn.Module):
    """行ごとに同じ全結合を掛ける。入力は (R, C)。"""

    def __init__(self, channels: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for cin, cout in zip(channels, channels[1:]):
            layers += [nn.Linear(cin, cout, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SetAbstractionMSG(nn.Module):
    def __init__(self, npoint: int, radii: list[float], nsamples: list[int], mlps: list[list[int]], in_channels: int):
        super().__init__()
        if not (len(radii) == len(nsamples) == len(mlps)):
            raise ValueError("radii、nsamples、mlps の長さをそろえてください。")
        self.npoint = npoint
        self.radii = list(radii)
        self.nsamples = list(nsamples)
        self.branches = nn.ModuleList(_SharedMLP([in_channels + 3, *mlp]) for mlp in mlps)
        self.out_channels = sum(mlp[-1] for mlp in mlps)

    def forward(
        self, xyz: torch.Tensor, feat: torch.Tensor, mask: torch.Tensor, random_start: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, _ = xyz.shape
        npoint = min(self.npoint, count)
        new_xyz = gather_points(xyz, farthest_point_sample(xyz, mask, npoint, random_start))
        counts = mask.sum(dim=1, keepdim=True)
        new_mask = torch.arange(npoint, device=xyz.device).unsqueeze(0) < counts
        outs = []
        for radius, k, mlp in zip(self.radii, self.nsamples, self.branches):
            idx, valid = ball_query(new_xyz, new_mask, xyz, mask, radius, k)
            rel = (gather_points(xyz, idx) - new_xyz.unsqueeze(2)) / radius
            grouped = torch.cat([rel, gather_points(feat, idx)], dim=-1)
            hidden = grouped.new_full((batch, npoint, k, mlp.out_channels), float("-inf"))
            hidden[valid] = mlp(grouped[valid]).to(hidden.dtype)
            pooled = hidden.max(dim=2).values
            outs.append(torch.where(new_mask.unsqueeze(-1), pooled, torch.zeros_like(pooled)))
        return new_xyz, torch.cat(outs, dim=-1), new_mask


class PointNet2Encoder(PointEncoder):
    """セット抽象化を重ね、最後に全点（ロボット座標の位置つき）を max でまとめる。

    近傍の相対位置は半径で割り、最後の段の絶対位置は coord_scale を掛けて 1 前後にそろえる。
    """

    def __init__(
        self,
        in_channels: int,
        sa_layers: list[dict] | tuple[dict, ...] = DEFAULT_SA_LAYERS,
        global_mlp: list[int] | tuple[int, ...] = (256, 512),
        coord_scale: float = 1.0 / 8.0,
    ):
        super().__init__()
        self.coord_scale = coord_scale
        self.layers = nn.ModuleList()
        channels = in_channels
        for layer in sa_layers:
            sa = _SetAbstractionMSG(layer["npoint"], layer["radii"], layer["nsamples"], layer["mlps"], channels)
            self.layers.append(sa)
            channels = sa.out_channels
        self.global_mlp = _SharedMLP([channels + 3, *global_mlp])
        self.out_channels = self.global_mlp.out_channels

    def forward(self, points: PointBatch) -> torch.Tensor:
        xyz, mask = to_padded(points.coord, points.batch, points.batch_size)
        feat, _ = to_padded(points.feat, points.batch, points.batch_size)
        for layer in self.layers:
            xyz, feat, mask = layer(xyz, feat, mask, random_start=self.training)
        grouped = torch.cat([xyz * self.coord_scale, feat], dim=-1)
        hidden = grouped.new_full((*mask.shape, self.out_channels), float("-inf"))
        hidden[mask] = self.global_mlp(grouped[mask]).to(hidden.dtype)
        return hidden.max(dim=1).values
