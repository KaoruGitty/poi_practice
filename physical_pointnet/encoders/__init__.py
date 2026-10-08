"""点群エンコーダの登録。名前で選び、設定は辞書でそのまま渡す。

このブランチの既定バックボーンは OSS を import する版:
- pointnet2 → yanx27/Pointnet_Pointnet2_pytorch
- point_transformer → qq456cvb/Point-Transformers（Hengshuang）

自前実装は pointnet2_native / ptv3 として残してある。
"""

from typing import Dict, Type

from .base import PointEncoder
from .point_transformer import PointTransformerEncoder
from .pointnet2 import PointNet2Encoder
from .pointnet2_native import PointNet2NativeEncoder
from .ptv3 import PTv3Encoder

ENCODERS = {
    "pointnet2": PointNet2Encoder,
    "point_transformer": PointTransformerEncoder,
    "pointnet2_native": PointNet2NativeEncoder,
    "ptv3": PTv3Encoder,
}  # type: Dict[str, Type[PointEncoder]]


def build_encoder(name, in_channels, **kwargs):
    # type: (str, int, **object) -> PointEncoder
    if name not in ENCODERS:
        raise ValueError(
            "エンコーダは {0} のどれかです。今は {1!r} です。".format(
                ", ".join(ENCODERS), name
            )
        )
    return ENCODERS[name](in_channels=in_channels, **kwargs)


__all__ = [
    "ENCODERS",
    "PointEncoder",
    "PointNet2Encoder",
    "PointTransformerEncoder",
    "PointNet2NativeEncoder",
    "PTv3Encoder",
    "build_encoder",
]
