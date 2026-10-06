"""方針ネットワーク。点群エンコーダ、状態の枝、回帰ヘッドを別々のモジュールとして組み合わせる。

    点群 ── エンコーダ（PointNet++ か PTv3）──┐
                                              ├─ 連結 ── 回帰ヘッド ── v, ω
    状態 ── 状態の枝（全結合）───────────────┘

エンコーダは名前と設定の辞書で選ぶ。状態の枝と回帰ヘッドはどのエンコーダでも共通。
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .config import SimConfig
from .encoders import build_encoder
from .points import PointBatch
from .preprocess import PreprocessConfig

CHECKPOINT_VERSION = 2

@dataclass
class ModelConfig:
    encoder: str = "pointnet2"
    encoder_kwargs: dict = field(default_factory=dict)
    in_channels: int = 3
    status_dim: int = 5
    use_status: bool = True
    status_hidden: Tuple[int, ...] = (64, 64)
    head_hidden: Tuple[int, ...] = (128,)
    dropout: float = 0.2
    out_dim: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "ModelConfig":
        data = dict(raw)
        for key in ("status_hidden", "head_hidden"):
            if key in data:
                data[key] = tuple(data[key])
        return ModelConfig(**data)

def _mlp(channels: List[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for cin, cout in zip(channels, channels[1:]):
        layers += [nn.Linear(cin, cout), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class StatusEncoder(nn.Module):
    """身体状態（速度、目標位置など）をベクトルにする。"""

    def __init__(self, status_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        self.net = _mlp([status_dim, *hidden])
        self.out_channels = hidden[-1] if hidden else status_dim

    def forward(self, status: torch.Tensor) -> torch.Tensor:
        return self.net(status)

class RegressionHead(nn.Module):
    """連結したベクトルから連続値を出す。最後の層に活性化は付けない。"""

    def __init__(self, in_channels: int, hidden: Tuple[int, ...], out_dim: int, dropout: float):
        super().__init__()
        self.body = _mlp([in_channels, *hidden], dropout)
        self.out = nn.Linear(hidden[-1] if hidden else in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))

class PolicyModel(nn.Module):
    def __init__(self, encoder: nn.Module, status_encoder: Optional[StatusEncoder], head: RegressionHead):
        super().__init__()
        self.encoder = encoder
        self.status_encoder = status_encoder
        self.head = head

    def forward(self, points: PointBatch, status: torch.Tensor) -> torch.Tensor:
        z = self.encoder(points)
        if self.status_encoder is not None:
            z = torch.cat([z, self.status_encoder(status).to(z.dtype)], dim=1)
        return self.head(z)

def build_model(cfg: ModelConfig) -> PolicyModel:
    encoder = build_encoder(cfg.encoder, cfg.in_channels, **cfg.encoder_kwargs)
    status_encoder = StatusEncoder(cfg.status_dim, cfg.status_hidden) if cfg.use_status else None
    width = encoder.out_channels + (status_encoder.out_channels if status_encoder is not None else 0)
    head = RegressionHead(width, cfg.head_hidden, cfg.out_dim, cfg.dropout)
    return PolicyModel(encoder, status_encoder, head)

def save_checkpoint(
    path: Union[str, Path],
    model: PolicyModel,
    model_cfg: ModelConfig,
    preprocess_cfg: PreprocessConfig,
    sim_cfg: SimConfig,
    **extra,
) -> None:
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model": model_cfg.to_dict(),
            "preprocess": preprocess_cfg.to_dict(),
            "sim": sim_cfg.to_dict(),
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )

def load_checkpoint(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> Tuple[PolicyModel, ModelConfig, PreprocessConfig, SimConfig, dict]:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} は旧形式のチェックポイントです。エンコーダとヘッドを分けた今の実装では読み込めません。"
            "学習し直してください。"
        )
    model_cfg = ModelConfig.from_dict(ckpt["model"])
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, model_cfg, PreprocessConfig.from_dict(ckpt["preprocess"]), SimConfig.from_dict(ckpt["sim"]), ckpt
