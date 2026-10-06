"""ナビゲーション環境。観測は多層LiDAR点群と身体状態。"""

from typing import Optional, Tuple

import math

import numpy as np

from ..config import SimConfig
from ..obs import make_status
from .geometry import body_dirs, collides, raycast
from .maps import World
from .robot import RobotState, step_robot

class NavEnv:
    def __init__(self, world: World, cfg: SimConfig, rng: np.random.Generator):
        self.world = world
        self.cfg = cfg
        self.rng = rng
        self.dirs_body = body_dirs(cfg.n_azimuth, cfg.elevations_deg)
        self.state = RobotState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self._points: Optional[np.ndarray] = None

    def reset(self) -> None:
        cfg = self.cfg
        world = self.world
        obstacles = list(world.obstacles)
        spawn_r = cfg.robot_radius + 0.2
        start = self._sample_xy(world.start_region, obstacles, spawn_r)
        goal = start
        for _ in range(80):
            goal = self._sample_xy(world.goal_region, obstacles, spawn_r)
            if math.hypot(goal[0] - start[0], goal[1] - start[1]) >= 3.5:
                break
        if self.rng.random() < 0.7:
            yaw = math.atan2(goal[1] - start[1], goal[0] - start[0]) + float(self.rng.normal(0.0, 0.35))
        else:
            yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.state = RobotState(start[0], start[1], yaw, 0.0, 0.0)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.steps = 0
        self._points = None

    def _sample_xy(
        self,
        region: Tuple[float, float, float, float],
        obstacles: list,
        radius: float,
    ) -> np.ndarray:
        xmin, xmax, ymin, ymax = region
        for _ in range(300):
            x = float(self.rng.uniform(xmin, xmax))
            y = float(self.rng.uniform(ymin, ymax))
            if not collides(x, y, radius, obstacles, self.world.width, self.world.height):
                return np.array([x, y], dtype=np.float32)
        raise RuntimeError(f"{self.world.name} で自由な座標を引けませんでした")

    def goal_distance(self) -> float:
        return float(math.hypot(self.goal[0] - self.state.x, self.goal[1] - self.state.y))

    def reached(self) -> bool:
        return self.goal_distance() <= self.cfg.goal_radius

    def observe(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._points is None:
            self._points = self._scan()
        status = make_status(self.state, self.goal, self.cfg)
        return self._points, status

    def _scan(self) -> np.ndarray:
        state = self.state
        dirs_body = self.dirs_body
        c = math.cos(state.yaw)
        s = math.sin(state.yaw)
        xb = dirs_body[:, 0]
        yb = dirs_body[:, 1]
        dirs_world = np.stack([c * xb - s * yb, s * xb + c * yb, dirs_body[:, 2]], axis=1)
        origin = np.array([state.x, state.y, self.cfg.sensor_z], dtype=np.float64)
        distances = raycast(origin, dirs_world, self.world.solids(), self.cfg.lidar_max_range)
        points = dirs_body * distances[:, None]
        points[:, 2] += np.float32(self.cfg.sensor_z)
        return points.astype(np.float32)

    def step(self, v_cmd: float, w_cmd: float) -> bool:
        self.state, collision = step_robot(self.state, v_cmd, w_cmd, self.world, self.cfg)
        self.steps += 1
        self._points = None
        return collision
