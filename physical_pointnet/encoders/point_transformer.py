"""Point Transformer エンコーダ（OSS import 版）。

qq456cvb/Point-Transformers の Hengshuang Zhao 実装（ICCV 2021）を
``third_party.point_transformer`` から import し、分類ヘッドの手前の
backbone 出力を平均して回帰用ベクトルにする。

点数は公開実装に合わせて固定長へ揃える（``num_point``）。
"""

import torch
from torch import nn

from third_party.point_transformer import Backbone

from ..points import PointBatch
from .base import PointEncoder
from .oss_io import pointbatch_to_fixed


class PointTransformerEncoder(PointEncoder):
    """Hengshuang Point Transformer backbone + 平均プーリング。"""

    def __init__(
        self,
        in_channels,  # type: int
        num_point=256,  # type: int
        nblocks=2,  # type: int
        nneighbor=16,  # type: int
        transformer_dim=64,  # type: int
        pool="mean",  # type: str
    ):
        # type: (...) -> None
        super(PointTransformerEncoder, self).__init__()
        if pool not in ("mean", "max"):
            raise ValueError("pool は mean か max です。今は {!r} です。".format(pool))
        self.num_point = int(num_point)
        self.pool = pool
        # Backbone は x[..., :3] をメートル座標として使うので、先頭に coord を付ける。
        input_dim = 3 + int(in_channels)
        self.backbone = Backbone(
            num_point=self.num_point,
            nblocks=int(nblocks),
            nneighbor=int(nneighbor),
            input_dim=input_dim,
            transformer_dim=int(transformer_dim),
        )
        self.out_channels = self.backbone.out_channels

    def forward(self, points):
        # type: (PointBatch) -> torch.Tensor
        xyz_bn3, feat_bnc = pointbatch_to_fixed(points, self.num_point)
        x = torch.cat([xyz_bn3, feat_bnc], dim=-1)
        feat = self.backbone(x)  # (B, S, C)
        if self.pool == "max":
            return feat.max(dim=1)[0]
        return feat.mean(dim=1)
