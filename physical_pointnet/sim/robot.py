"""加速度制限つきの差動二輪。現在速度が次の動きを決める。"""

from typing import List, Tuple

import math
from dataclasses import dataclass

from ..config import SimConfig
from .geometry import AABB, collides, wrap_angle
from .maps import World


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    w: float


def step_robot(
    state: RobotState,
    v_cmd: float,
    w_cmd: float,
    world: World,
    cfg: SimConfig,
) -> Tuple[RobotState, bool]:
    """1ステップ進める。めり込む場合は軸ごとに滑らせ、両方ダメなら衝突。"""
    v_cmd = min(max(v_cmd, 0.0), cfg.v_max)
    w_cmd = min(max(w_cmd, -cfg.w_max), cfg.w_max)
    dv = min(max(v_cmd - state.v, -cfg.a_max * cfg.dt), cfg.a_max * cfg.dt)
    dw = min(max(w_cmd - state.w, -cfg.alpha_max * cfg.dt), cfg.alpha_max * cfg.dt)
    v = min(max(state.v + dv, 0.0), cfg.v_max)
    w = min(max(state.w + dw, -cfg.w_max), cfg.w_max)
    yaw = wrap_angle(state.yaw + w * cfg.dt)
    nx = state.x + v * math.cos(state.yaw) * cfg.dt
    ny = state.y + v * math.sin(state.yaw) * cfg.dt
    obstacles: List[AABB] = list(world.obstacles)
    if not collides(nx, ny, cfg.robot_radius, obstacles, world.width, world.height):
        return RobotState(nx, ny, yaw, v, w), False
    slide_x = not collides(nx, state.y, cfg.robot_radius, obstacles, world.width, world.height)
    slide_y = not collides(state.x, ny, cfg.robot_radius, obstacles, world.width, world.height)
    if slide_x:
        return RobotState(nx, state.y, yaw, 0.0, w), False
    if slide_y:
        return RobotState(state.x, ny, yaw, 0.0, w), False
    return RobotState(state.x, state.y, yaw, 0.0, w), True
