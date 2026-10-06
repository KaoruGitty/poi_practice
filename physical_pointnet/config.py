"""シミュレータと観測のスケール。収集・学習・評価で共有する。"""

from __future__ import annotations

from typing import Tuple

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.1
    v_max: float = 1.0
    w_max: float = 1.5
    a_max: float = 0.8
    alpha_max: float = 2.2
    robot_radius: float = 0.30
    sensor_z: float = 0.50
    lidar_max_range: float = 8.0
    z_scale: float = 2.0
    n_azimuth: int = 72
    elevations_deg: Tuple[float, ...] = (-12.0, -2.0, 3.0, 10.0)
    goal_radius: float = 0.45
    max_steps: int = 420
    goal_scale: float = 12.0
    status_dim: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "SimConfig":
        data = dict(raw)
        if "elevations_deg" in data:
            data["elevations_deg"] = tuple(data["elevations_deg"])
        return SimConfig(**data)
