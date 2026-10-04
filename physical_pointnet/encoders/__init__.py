"""点群エンコーダの登録。名前で選び、設定は辞書でそのまま渡す。"""

from __future__ import annotations

from .base import PointEncoder
from .pointnet2 import PointNet2Encoder
from .ptv3 import PTv3Encoder

ENCODERS: dict[str, type[PointEncoder]] = {
    "pointnet2": PointNet2Encoder,
    "ptv3": PTv3Encoder,
}


def build_encoder(name: str, in_channels: int, **kwargs) -> PointEncoder:
    if name not in ENCODERS:
        raise ValueError(f"エンコーダは {', '.join(ENCODERS)} のどれかです。今は {name!r} です。")
    return ENCODERS[name](in_channels=in_channels, **kwargs)


__all__ = ["ENCODERS", "PointEncoder", "PointNet2Encoder", "PTv3Encoder", "build_encoder"]
