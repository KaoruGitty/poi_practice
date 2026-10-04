"""LiDAR の 1 スキャンをエンコーダの入力にそろえる。学習と推論で同じ関数を通す。

入力は (N, 3 + E) の配列で、先頭 3 列が xyz（メートル）、残りが extra_channels の順の追加列。
出力は、ロボット座標のメートルの座標 (n, 3) と、点ごとの特徴 (n, 3 + E)。
点数は毎回変わってよい。バッチにするときは points.PointBatch で縦につなぐ。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import SimConfig

_NO_RETURN_TOL = 1e-4


@dataclass(frozen=True)
class PreprocessConfig:
    # 距離はすべて range_origin（入力座標でのセンサ位置）からの 3 次元距離。
    max_range: float = 8.0
    min_range: float = 0.0
    range_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 距離が max_range ちょうどの点を「返ってこなかったビーム」として落とす。
    drop_no_return: bool = True
    # 入力座標からロボット座標への変換。回転は行優先の 3x3。
    sensor_rotation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    sensor_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ロボット座標での高さの範囲。
    z_min: float = -1.0
    z_max: float = 3.0
    # ボクセルごとに 1 点だけ残す。0 なら間引かない。
    voxel_size: float = 0.0
    # 点数の上限。0 なら上限なし。超えたら random か fps で減らす。
    max_points: int = 0
    sampling: str = "random"
    # 特徴に入れる座標の倍率。値が 1 前後に収まるようにする。
    coord_scale_xy: float = 1.0 / 8.0
    coord_scale_z: float = 0.5
    extra_channels: tuple[str, ...] = ()
    extra_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.extra_channels) != len(self.extra_scales):
            raise ValueError("extra_channels と extra_scales の長さをそろえてください。")
        if self.sampling not in ("random", "fps"):
            raise ValueError("sampling は random か fps です。")

    @property
    def in_channels(self) -> int:
        return 3 + len(self.extra_channels)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "PreprocessConfig":
        data = dict(raw)
        for key in ("range_origin", "sensor_rotation", "sensor_translation", "extra_channels", "extra_scales"):
            if key in data:
                data[key] = tuple(data[key])
        return PreprocessConfig(**data)

    @staticmethod
    def for_sim(cfg: SimConfig) -> "PreprocessConfig":
        """このリポジトリのシミュレータ用。点はロボット座標で、センサは床から sensor_z の高さ。"""
        return PreprocessConfig(
            max_range=cfg.lidar_max_range,
            range_origin=(0.0, 0.0, cfg.sensor_z),
            z_min=-0.5,
            z_max=3.0,
            coord_scale_xy=1.0 / cfg.lidar_max_range,
            coord_scale_z=1.0 / cfg.z_scale,
        )


@dataclass(frozen=True)
class AugmentConfig:
    """学習時だけの増強。推論では使わない。"""

    jitter_std: float = 0.01  # m
    jitter_clip: float = 0.05  # m
    drop_ratio: float = 0.1
    # 左右反転。ラベル側（角速度、目標の左右位置）の符号は dataset が反転する。
    mirror_prob: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def voxel_downsample(xyz: np.ndarray, voxel: float, rng: np.random.Generator | None) -> np.ndarray:
    """ボクセルごとに 1 点の番号を返す。rng があればランダム、なければボクセル中心に最も近い点。"""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if rng is not None:
        tie = rng.random(len(xyz))
    else:
        tie = np.linalg.norm(xyz - (keys + 0.5) * voxel, axis=1)
    order = np.lexsort((tie, inverse))
    grouped = inverse[order]
    first = np.r_[True, grouped[1:] != grouped[:-1]]
    return np.sort(order[first])


def farthest_point_indices(xyz: np.ndarray, n: int, rng: np.random.Generator | None) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    dist = np.full(len(xyz), np.inf)
    idx[0] = int(rng.integers(len(xyz))) if rng is not None else 0
    for i in range(1, n):
        dist = np.minimum(dist, ((xyz - xyz[idx[i - 1]]) ** 2).sum(axis=1))
        idx[i] = int(dist.argmax())
    return np.sort(idx)


def preprocess_scan(
    points: np.ndarray,
    cfg: PreprocessConfig,
    rng: np.random.Generator | None = None,
    augment: AugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """1 スキャンを (座標, 特徴, 左右反転したか) にする。

    rng を渡すとボクセルの代表点と点数の削減がランダムになる。augment は学習時だけ渡す。
    有効な点が 1 つも残らないときは、ロボット原点に特徴 0 の点を 1 つだけ置く。
    """
    extra_dim = len(cfg.extra_channels)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 + extra_dim:
        raise ValueError(f"点群は (N, {3 + extra_dim}) 以上の形にしてください。今は {pts.shape} です。")
    xyz = pts[:, :3].astype(np.float64)
    extra = pts[:, 3 : 3 + extra_dim].astype(np.float64)

    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(extra).all(axis=1)
    rng_dist = np.linalg.norm(xyz - np.asarray(cfg.range_origin), axis=1)
    keep &= rng_dist >= cfg.min_range
    if cfg.drop_no_return:
        keep &= rng_dist < cfg.max_range * (1.0 - _NO_RETURN_TOL)
    else:
        keep &= rng_dist <= cfg.max_range
    xyz, extra = xyz[keep], extra[keep]

    rotation = np.asarray(cfg.sensor_rotation, dtype=np.float64).reshape(3, 3)
    xyz = xyz @ rotation.T + np.asarray(cfg.sensor_translation)
    keep = (xyz[:, 2] >= cfg.z_min) & (xyz[:, 2] <= cfg.z_max)
    xyz, extra = xyz[keep], extra[keep]

    mirrored = False
    if augment is not None and rng is not None and len(xyz):
        if rng.random() < augment.mirror_prob:
            xyz[:, 1] = -xyz[:, 1]
            mirrored = True
        if augment.jitter_std > 0:
            noise = rng.normal(0.0, augment.jitter_std, size=xyz.shape)
            xyz = xyz + np.clip(noise, -augment.jitter_clip, augment.jitter_clip)
        if augment.drop_ratio > 0 and len(xyz) > 1:
            keep = rng.random(len(xyz)) >= augment.drop_ratio
            if keep.any():
                xyz, extra = xyz[keep], extra[keep]

    if cfg.voxel_size > 0 and len(xyz):
        idx = voxel_downsample(xyz, cfg.voxel_size, rng)
        xyz, extra = xyz[idx], extra[idx]

    if cfg.max_points > 0 and len(xyz) > cfg.max_points:
        if cfg.sampling == "fps":
            idx = farthest_point_indices(xyz, cfg.max_points, rng)
        else:
            chooser = rng if rng is not None else np.random.default_rng(0)
            idx = np.sort(chooser.choice(len(xyz), cfg.max_points, replace=False))
        xyz, extra = xyz[idx], extra[idx]

    if len(xyz) == 0:
        coord = np.zeros((1, 3), dtype=np.float32)
        return coord, np.zeros((1, cfg.in_channels), dtype=np.float32), mirrored

    scale = np.array([cfg.coord_scale_xy, cfg.coord_scale_xy, cfg.coord_scale_z])
    feat = [xyz * scale]
    if extra_dim:
        feat.append(extra * np.asarray(cfg.extra_scales))
    return xyz.astype(np.float32), np.concatenate(feat, axis=1).astype(np.float32), mirrored
