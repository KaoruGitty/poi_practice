"""状態と行動のスケール。学習と閉ループで同じ式を使う。点群の前処理は preprocess.py。"""

from typing import Tuple

import math

import numpy as np

from .config import SimConfig
from .sim.robot import RobotState


def make_status(state: RobotState, goal: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """身体速度と、ロボット座標で表した目標位置。"""
    dx = float(goal[0] - state.x)
    dy = float(goal[1] - state.y)
    c = math.cos(state.yaw)
    s = math.sin(state.yaw)
    goal_x = c * dx + s * dy
    goal_y = -s * dx + c * dy
    goal_dist = math.hypot(goal_x, goal_y)
    return np.array(
        [
            state.v / cfg.v_max,
            state.w / cfg.w_max,
            goal_x / cfg.goal_scale,
            goal_y / cfg.goal_scale,
            goal_dist / cfg.goal_scale,
        ],
        dtype=np.float32,
    )


# 左右反転したときに符号が変わる成分。状態は v, ω, 目標の前後, 目標の左右, 距離。行動は v, ω。
STATUS_MIRROR_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
ACTION_MIRROR_SIGN = np.array([1.0, -1.0], dtype=np.float32)


def normalize_action(v: float, w: float, cfg: SimConfig) -> np.ndarray:
    return np.array([v / cfg.v_max, w / cfg.w_max], dtype=np.float32)


def denormalize_action(pred: np.ndarray, cfg: SimConfig) -> Tuple[float, float]:
    v = float(np.clip(pred[0], 0.0, 1.0) * cfg.v_max)
    w = float(np.clip(pred[1], -1.0, 1.0) * cfg.w_max)
    return v, w
