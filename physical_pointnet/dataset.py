"""教師デモの保存と読み込み。エピソード単位で学習と検証を分ける。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SimConfig
from .obs import ACTION_MIRROR_SIGN, STATUS_MIRROR_SIGN
from .points import PointBatch
from .preprocess import AugmentConfig, PreprocessConfig, preprocess_scan


def save_demos(path: str | Path, payload: dict[str, np.ndarray], cfg: SimConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(payload)
    arrays["v_max"] = np.float32(cfg.v_max)
    arrays["w_max"] = np.float32(cfg.w_max)
    arrays["lidar_max_range"] = np.float32(cfg.lidar_max_range)
    arrays["z_scale"] = np.float32(cfg.z_scale)
    np.savez_compressed(path, **arrays)


def load_demos(path: str | Path) -> dict[str, np.ndarray]:
    """点数がフレームごとに違う points はオブジェクト配列で保存されるので、リストに戻す。"""
    with np.load(path, allow_pickle=True) as data:
        demos = {key: data[key] for key in data.files}
    if demos["points"].dtype == object:
        demos["points"] = list(demos["points"])
    return demos


def split_episodes(episode_ids: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    order = np.random.default_rng(seed).permutation(episodes)
    n_val = max(1, int(round(len(order) * val_ratio)))
    val_eps = set(order[:n_val].tolist())
    train_idx = np.array([i for i, ep in enumerate(episode_ids) if int(ep) not in val_eps], dtype=np.int64)
    val_idx = np.array([i for i, ep in enumerate(episode_ids) if int(ep) in val_eps], dtype=np.int64)
    return train_idx, val_idx


class DemoDataset(Dataset):
    """1 サンプルずつ前処理して返す。点数はサンプルごとに変わる。

    points は (M, N, 3 + E) の配列か、(N_i, 3 + E) の配列のリスト。
    """

    def __init__(
        self,
        demos: dict[str, np.ndarray],
        indices: np.ndarray,
        preprocess: PreprocessConfig,
        augment: AugmentConfig | None = None,
    ):
        points = demos["points"]
        self.points = [points[i] for i in indices] if isinstance(points, list) else points[indices]
        self.status = demos["status"][indices]
        self.action = demos["action"][indices]
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self) -> int:
        return len(self.status)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        rng = None
        if self.augment is not None:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        coord, feat, mirrored = preprocess_scan(self.points[index], self.preprocess, rng, self.augment)
        status = self.status[index].astype(np.float32)
        action = self.action[index].astype(np.float32)
        if mirrored:
            status = status * STATUS_MIRROR_SIGN
            action = action * ACTION_MIRROR_SIGN
        return {"coord": coord, "feat": feat, "status": status, "action": action}


def collate_samples(samples: list[dict[str, np.ndarray]]) -> tuple[PointBatch, torch.Tensor, torch.Tensor]:
    points = PointBatch.from_arrays([s["coord"] for s in samples], [s["feat"] for s in samples])
    status = torch.from_numpy(np.stack([s["status"] for s in samples]))
    action = torch.from_numpy(np.stack([s["action"] for s in samples]))
    return points, status, action
