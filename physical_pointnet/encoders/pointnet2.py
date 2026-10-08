"""PointNet++ エンコーダ（OSS import 版）。

yanx27/Pointnet_Pointnet2_pytorch の ``PointNetSetAbstractionMsg`` /
``PointNetSetAbstraction`` を ``third_party.pointnet2_pytorch`` から import し、
分類ヘッドの手前までを回帰用のベクトルエンコーダとして使う。

点数は公開実装に合わせて固定長へ揃える（``num_point``）。半径はメートル。
"""

from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

from third_party.pointnet2_pytorch import PointNetSetAbstraction, PointNetSetAbstractionMsg

from ..points import PointBatch
from .base import PointEncoder
from .oss_io import pointbatch_to_fixed

# このリポジトリのシミュレータ（1 スキャン 300 点弱、最大 8 m）に合わせた既定値。
DEFAULT_SA_LAYERS = (
    {"npoint": 64, "radii": [0.8, 1.6], "nsamples": [8, 16], "mlps": [[32, 32, 64], [64, 64, 128]]},
    {"npoint": 16, "radii": [2.0, 4.0], "nsamples": [8, 16], "mlps": [[64, 64, 128], [128, 128, 256]]},
)


class PointNet2Encoder(PointEncoder):
    """OSS のセット抽象化を重ね、最後に group_all で 1 本のベクトルにする。"""

    def __init__(
        self,
        in_channels,  # type: int
        sa_layers=DEFAULT_SA_LAYERS,
        global_mlp=(256, 512),
        num_point=256,
        use_feature=True,
    ):
        # type: (int, Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...]], Union[List[int], Tuple[int, ...]], int, bool) -> None
        super(PointNet2Encoder, self).__init__()
        self.num_point = int(num_point)
        self.use_feature = bool(use_feature)
        layers = list(sa_layers)
        if not layers:
            raise ValueError("sa_layers が空です。")

        feat_channels = in_channels if self.use_feature else 0
        self.sa_modules = nn.ModuleList()
        channels = feat_channels
        for layer in layers:
            sa = PointNetSetAbstractionMsg(
                layer["npoint"],
                list(layer["radii"]),
                list(layer["nsamples"]),
                channels,
                [list(m) for m in layer["mlps"]],
            )
            self.sa_modules.append(sa)
            channels = sum(m[-1] for m in layer["mlps"])

        gmlp = list(global_mlp)
        self.sa_global = PointNetSetAbstraction(
            None, None, None, channels + 3, gmlp, group_all=True
        )
        self.out_channels = gmlp[-1]

    def forward(self, points):
        # type: (PointBatch) -> torch.Tensor
        xyz_bn3, feat_bnc = pointbatch_to_fixed(points, self.num_point)
        xyz = xyz_bn3.transpose(1, 2).contiguous()
        if self.use_feature:
            feat = feat_bnc.transpose(1, 2).contiguous()
        else:
            feat = None
        for sa in self.sa_modules:
            xyz, feat = sa(xyz, feat)
        xyz, feat = self.sa_global(xyz, feat)
        return feat.squeeze(-1)
