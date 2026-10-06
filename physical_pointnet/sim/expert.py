"""幾何を見られる教師。学生モデルは点群と状態しか見ない。

障害物をロボット半径ぶん膨らませた格子で A* を引き、
その経路を pure pursuit で追う。曲率が大きいところと壁の近くでは減速する。
経路が引けないときだけ、見出し角をサンプルする局所回避に戻す。
"""

from typing import Optional, Dict, List, Tuple

import heapq
import math

import numpy as np

from ..config import SimConfig
from .geometry import clearance, collides, wrap_angle
from .maps import World
from .robot import RobotState

_GRID_CACHE: Dict[str, Tuple[float, np.ndarray]] = {}

def _occupancy(world: World, cfg: SimConfig) -> Tuple[float, np.ndarray]:
    cached = _GRID_CACHE.get(world.name)
    if cached is not None:
        return cached
    resolution = 0.25
    radius = cfg.robot_radius + 0.40
    xs = np.arange(resolution / 2.0, world.width, resolution)
    ys = np.arange(resolution / 2.0, world.height, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    blocked = (
        (x_grid < radius)
        | (y_grid < radius)
        | (x_grid > world.width - radius)
        | (y_grid > world.height - radius)
    )
    for box in world.obstacles:
        dx = np.maximum(np.maximum(box.xmin - x_grid, 0.0), x_grid - box.xmax)
        dy = np.maximum(np.maximum(box.ymin - y_grid, 0.0), y_grid - box.ymax)
        blocked |= dx * dx + dy * dy < radius * radius
    _GRID_CACHE[world.name] = (resolution, blocked)
    return resolution, blocked

def _cell(value: float, resolution: float, limit: int) -> int:
    index = int(round((value - resolution / 2.0) / resolution))
    return min(max(index, 0), limit - 1)

def _nearest_free(blocked: np.ndarray, iy: int, ix: int) -> Optional[Tuple[int, int]]:
    if not blocked[iy, ix]:
        return iy, ix
    h, w = blocked.shape
    for radius in range(1, max(h, w)):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < h and 0 <= nx < w and not blocked[ny, nx]:
                    return ny, nx
    return None

def _astar(blocked: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    height, width = blocked.shape
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    queue = [(heuristic(start, goal), 0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost = {start: 0.0}
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue:
        _, g_cost, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path
        if g_cost > cost.get(current, math.inf):
            continue
        for dy, dx in steps:
            ny, nx = current[0] + dy, current[1] + dx
            if not (0 <= ny < height and 0 <= nx < width) or blocked[ny, nx]:
                continue
            # 斜め移動で壁の角をすり抜けない。
            if dy and dx and (blocked[current[0] + dy, current[1]] or blocked[current[0], current[1] + dx]):
                continue
            step = math.sqrt(2.0) if dy and dx else 1.0
            next_cost = g_cost + step
            nxt = (ny, nx)
            if next_cost < cost.get(nxt, math.inf):
                cost[nxt] = next_cost
                came[nxt] = current
                heapq.heappush(queue, (next_cost + heuristic(nxt, goal), next_cost, nxt))
    return None

def _shortcut(points: np.ndarray, world: World, cfg: SimConfig, radius: float) -> np.ndarray:
    if len(points) < 3:
        return points
    kept = [points[0]]
    anchor = 0
    obstacles = list(world.obstacles)
    while anchor < len(points) - 1:
        nxt = len(points) - 1
        while nxt > anchor + 1:
            if _segment_free(points[anchor], points[nxt], obstacles, world, radius):
                break
            nxt -= 1
        kept.append(points[nxt])
        anchor = nxt
    return np.asarray(kept, dtype=np.float32)

def _segment_free(a: np.ndarray, b: np.ndarray, obstacles: list, world: World, radius: float) -> bool:
    dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    steps = max(int(dist / 0.2), 1)
    for i in range(1, steps):
        t = i / steps
        x = float(a[0] + (b[0] - a[0]) * t)
        y = float(a[1] + (b[1] - a[1]) * t)
        if collides(x, y, radius, obstacles, world.width, world.height):
            return False
    return True

def _plan_path(state: RobotState, goal: np.ndarray, world: World, cfg: SimConfig) -> Optional[np.ndarray]:
    resolution, blocked = _occupancy(world, cfg)
    h, w = blocked.shape
    start = _nearest_free(blocked, _cell(state.y, resolution, h), _cell(state.x, resolution, w))
    end = _nearest_free(blocked, _cell(float(goal[1]), resolution, h), _cell(float(goal[0]), resolution, w))
    if start is None or end is None:
        return None
    cells = _astar(blocked, start, end)
    if cells is None:
        return None
    centers = np.array(
        [[(ix + 0.5) * resolution, (iy + 0.5) * resolution] for iy, ix in cells],
        dtype=np.float32,
    )
    # meshgrid のセル中心は arange(resolution/2) なので (ix+0.5)*resolution と一致する。
    waypoints = np.vstack(
        [
            np.array([[state.x, state.y]], dtype=np.float32),
            centers,
            np.array([[goal[0], goal[1]]], dtype=np.float32),
        ]
    )
    return _shortcut(waypoints, world, cfg, cfg.robot_radius + 0.28)

def _clear(state: RobotState, heading: float, world: World, cfg: SimConfig, max_dist: float = 3.5) -> float:
    return clearance(
        state.x,
        state.y,
        heading,
        list(world.obstacles),
        world.width,
        world.height,
        cfg.robot_radius,
        max_dist,
    )

def _speed(goal_dist: float, front: float, err: float, cfg: SimConfig) -> float:
    if front < 0.16:
        return 0.0
    speed_scale = float(np.clip((front - 0.16) / 1.15, 0.0, 1.0))
    turn_scale = float(np.clip(1.0 - abs(err) / 1.15, 0.25, 1.0))
    arrive = float(np.clip((goal_dist - 0.05) / 1.6, 0.0, 1.0))
    floor = 0.18 if goal_dist > cfg.goal_radius else 0.0
    return cfg.v_max * speed_scale * turn_scale * max(arrive, floor)

def _pursuit(state: RobotState, path: np.ndarray, goal: np.ndarray, world: World, cfg: SimConfig) -> Tuple[float, float]:
    distance = np.hypot(path[:, 0] - state.x, path[:, 1] - state.y)
    nearest = int(np.argmin(distance))
    obstacles = list(world.obstacles)
    los_radius = cfg.robot_radius + 0.05
    travel = 0.0
    target_i = min(nearest + 1, len(path) - 1)
    lookahead = 0.85
    for i in range(nearest, len(path) - 1):
        if not _segment_free(np.array([state.x, state.y]), path[i + 1], obstacles, world, los_radius):
            break
        travel += float(np.hypot(path[i + 1, 0] - path[i, 0], path[i + 1, 1] - path[i, 1]))
        target_i = i + 1
        if travel >= lookahead:
            break
    target = path[target_i]
    desired = math.atan2(float(target[1] - state.y), float(target[0] - state.x))
    # 進路がふさがっているときは、経路方向の近くで空いている向きへ膨らむ。
    if _clear(state, desired, world, cfg) < 0.45:
        best_heading = desired
        best_score = -1e9
        for delta in np.linspace(-1.4, 1.4, 15):
            heading = desired + float(delta)
            free = _clear(state, heading, world, cfg)
            score = free - 0.7 * abs(wrap_angle(heading - desired))
            if score > best_score:
                best_score = score
                best_heading = heading
        desired = best_heading
    err = wrap_angle(desired - state.yaw)
    w_cmd = float(np.clip(2.6 * err, -cfg.w_max, cfg.w_max))
    goal_dist = math.hypot(float(goal[0] - state.x), float(goal[1] - state.y))
    return _speed(goal_dist, _clear(state, state.yaw, world, cfg), err, cfg), w_cmd

def _reactive(state: RobotState, goal: np.ndarray, world: World, cfg: SimConfig) -> Tuple[float, float]:
    dx = float(goal[0] - state.x)
    dy = float(goal[1] - state.y)
    goal_dist = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)
    headings = np.concatenate([state.yaw + np.linspace(-2.2, 2.2, 17), [goal_heading]])
    best_score = -1e9
    best_heading = state.yaw
    for heading in headings:
        free = clearance(
            state.x, state.y, float(heading), list(world.obstacles), world.width, world.height, cfg.robot_radius, 4.0
        )
        align = math.cos(float(heading) - goal_heading)
        turn = abs(wrap_angle(float(heading) - state.yaw))
        score = 1.7 * align + 0.22 * min(free, 4.0) - 0.28 * turn
        if free < 0.9:
            score -= (0.9 - free) * 3.5
        if score > best_score:
            best_score = score
            best_heading = float(heading)
    err = wrap_angle(best_heading - state.yaw)
    w_cmd = float(np.clip(2.4 * err, -cfg.w_max, cfg.w_max))
    front = clearance(
        state.x, state.y, state.yaw, list(world.obstacles), world.width, world.height, cfg.robot_radius, 4.0
    )
    if front < 0.18:
        return 0.0, w_cmd
    speed_scale = float(np.clip((front - 0.18) / 1.4, 0.0, 1.0))
    turn_scale = float(np.clip(1.0 - abs(err) / 1.4, 0.2, 1.0))
    arrive = float(np.clip((goal_dist - 0.1) / 1.8, 0.0, 1.0))
    return cfg.v_max * speed_scale * turn_scale * max(arrive, 0.12), w_cmd

def expert_action(state: RobotState, goal: np.ndarray, world: World, cfg: SimConfig) -> Tuple[float, float]:
    path = _plan_path(state, goal, world, cfg)
    if path is None or len(path) < 2:
        return _reactive(state, goal, world, cfg)
    return _pursuit(state, path, goal, world, cfg)
