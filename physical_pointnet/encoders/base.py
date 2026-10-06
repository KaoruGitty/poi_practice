"""点群エンコーダの共通の形。"""

import torch
from torch import nn

from ..points import PointBatch


class PointEncoder(nn.Module):
    """PointBatch を受け取り、サンプルごとに (B, out_channels) のベクトルを返す。"""

    out_channels: int

    def forward(self, points: PointBatch) -> torch.Tensor:  # pragma: no cover - 抽象
        raise NotImplementedError
