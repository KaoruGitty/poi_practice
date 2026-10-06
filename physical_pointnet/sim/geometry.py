"""軸平行ボックスに対するレイキャストと円衝突。"""

from typing import Optional, List, Tuple

import math
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class AABB:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @staticmethod
    def from_center(cx: float, cy: float, sx: float, sy: float, height: float) -> "AABB":
        return AABB(cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0, 0.0, height)

def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def _safe_inv(direction: np.ndarray) -> np.ndarray:
    out = np.empty_like(direction)
    small = np.abs(direction) < 1e-8
    out[~small] = 1.0 / direction[~small]
    signs = np.where(direction[small] == 0.0, 1.0, direction[small])
    out[small] = np.copysign(1e12, signs)
    return out

def raycast(
    origin: np.ndarray,
    dirs_world: np.ndarray,
    boxes: List[AABB],
    max_range: float,
) -> np.ndarray:
    """単位方向ベクトルごとの衝突距離を返す。地面・ボックス・最大距離の最短。"""
    ox, oy, oz = (float(v) for v in origin)
    dirs = np.asarray(dirs_world, dtype=np.float64)
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    n = dirs.shape[0]
    t_hit = np.full(n, max_range, dtype=np.float64)

    down = dz < -1e-8
    t_ground = np.full(n, np.inf, dtype=np.float64)
    t_ground[down] = (0.0 - oz) / dz[down]
    t_ground[t_ground < 1e-4] = np.inf
    t_hit = np.minimum(t_hit, t_ground)

    if boxes:
        bmin = np.array([[b.xmin, b.ymin, b.zmin] for b in boxes], dtype=np.float64)
        bmax = np.array([[b.xmax, b.ymax, b.zmax] for b in boxes], dtype=np.float64)
        components = (dx, dy, dz)
        origin_xyz = (ox, oy, oz)
        t_enter = np.full((len(boxes), n), -np.inf, dtype=np.float64)
        t_exit = np.full((len(boxes), n), np.inf, dtype=np.float64)
        for axis in range(3):
            inv_a = _safe_inv(components[axis])[None, :]
            o_a = origin_xyz[axis]
            b0 = bmin[:, axis][:, None]
            b1 = bmax[:, axis][:, None]
            t1 = (b0 - o_a) * inv_a
            t2 = (b1 - o_a) * inv_a
            tmin_a = np.minimum(t1, t2)
            tmax_a = np.maximum(t1, t2)
            small = np.abs(components[axis]) < 1e-8
            outside = small[None, :] & ((o_a < b0) | (o_a > b1))
            tmin_a = np.where(outside, np.inf, tmin_a)
            tmax_a = np.where(outside, -np.inf, tmax_a)
            t_enter = np.maximum(t_enter, tmin_a)
            t_exit = np.minimum(t_exit, tmax_a)
        valid = (t_exit >= np.maximum(t_enter, 1e-4)) & (t_exit > 1e-4)
        t_box = np.where(t_enter > 1e-4, t_enter, t_exit)
        t_box = np.where(valid, t_box, np.inf)
        t_hit = np.minimum(t_hit, t_box.min(axis=0))

    return np.clip(t_hit, 0.0, max_range).astype(np.float32)

def collides(x: float, y: float, radius: float, obstacles: List[AABB], width: float, height: float) -> bool:
    if x < radius or y < radius or x > width - radius or y > height - radius:
        return True
    r2 = radius * radius
    for box in obstacles:
        cx = min(max(x, box.xmin), box.xmax)
        cy = min(max(y, box.ymin), box.ymax)
        if (x - cx) ** 2 + (y - cy) ** 2 < r2:
            return True
    return False

def _ray_aabb_2d(
    x: float,
    y: float,
    dx: float,
    dy: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[float]:
    """原点がボックス外なら進入距離、内部なら 0、外れていれば None。"""
    ts_min = []
    ts_max = []
    for origin, direction, lo, hi in ((x, dx, xmin, xmax), (y, dy, ymin, ymax)):
        if abs(direction) < 1e-8:
            if origin < lo or origin > hi:
                return None
            ts_min.append(-math.inf)
            ts_max.append(math.inf)
            continue
        t1 = (lo - origin) / direction
        t2 = (hi - origin) / direction
        ts_min.append(min(t1, t2))
        ts_max.append(max(t1, t2))
    t_enter = max(ts_min)
    t_exit = min(ts_max)
    if t_exit < 0.0 or t_enter > t_exit:
        return None
    if t_enter > 0.0:
        return t_enter
    return 0.0

def clearance(
    x: float,
    y: float,
    yaw: float,
    obstacles: List[AABB],
    width: float,
    height: float,
    radius: float,
    max_dist: float,
) -> float:
    """ロボット中心が進める距離。障害物は半径ぶん膨らませて判定する。"""
    dx = math.cos(yaw)
    dy = math.sin(yaw)
    best = max_dist
    if dx > 1e-8:
        best = min(best, (width - radius - x) / dx)
    elif dx < -1e-8:
        best = min(best, (radius - x) / dx)
    if dy > 1e-8:
        best = min(best, (height - radius - y) / dy)
    elif dy < -1e-8:
        best = min(best, (radius - y) / dy)
    for box in obstacles:
        hit = _ray_aabb_2d(
            x,
            y,
            dx,
            dy,
            box.xmin - radius,
            box.xmax + radius,
            box.ymin - radius,
            box.ymax + radius,
        )
        if hit is not None:
            best = min(best, hit)
    return float(max(0.0, best))

def body_dirs(n_azimuth: int, elevations_deg: Tuple[float, ...]) -> np.ndarray:
    azimuth = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
    elevation = np.deg2rad(np.asarray(elevations_deg, dtype=np.float64))
    az_grid, el_grid = np.meshgrid(azimuth, elevation, indexing="xy")
    cos_el = np.cos(el_grid)
    dirs = np.stack(
        [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
        axis=-1,
    )
    return dirs.reshape(-1, 3).astype(np.float32)

def to_world(points_body: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    xb = points_body[:, 0]
    yb = points_body[:, 1]
    out = np.empty_like(points_body)
    out[:, 0] = x + c * xb - s * yb
    out[:, 1] = y + s * xb + c * yb
    out[:, 2] = points_body[:, 2]
    return out

def selftest() -> None:
    tall = AABB.from_center(5.0, 2.0, 1.0, 1.0, 1.5)
    origin = np.array([2.0, 2.0, 0.5], dtype=np.float64)
    forward = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    t_forward = raycast(origin, forward, [tall], 8.0)
    if abs(float(t_forward[0]) - 2.5) > 1e-3:
        raise AssertionError(f"前方レイが側面に当たらない: {t_forward[0]}")

    low = AABB.from_center(5.0, 2.0, 1.0, 1.0, 0.3)
    t_miss = raycast(origin, forward, [low], 8.0)
    if float(t_miss[0]) < 7.9:
        raise AssertionError(f"水平ビームが低い障害物に当たっている: {t_miss[0]}")

    downward = np.array([[2.5, 0.0, -0.25]], dtype=np.float64)
    downward /= np.linalg.norm(downward, axis=1, keepdims=True)
    t_down = raycast(origin, downward, [low], 8.0)
    if float(t_down[0]) > 4.0:
        raise AssertionError(f"下向きビームが低い障害物を見落としている: {t_down[0]}")

    if not collides(4.7, 2.0, 0.3, [tall], 14.0, 10.0):
        raise AssertionError("円がボックスにめり込んでいるのに衝突しない")
    if collides(3.5, 2.0, 0.3, [tall], 14.0, 10.0):
        raise AssertionError("離れた円を衝突と判定した")

    gap = clearance(2.0, 2.0, 0.0, [tall], 14.0, 10.0, 0.3, 8.0)
    # 側面 x=4.5 を半径 0.3 だけ手前で止めるので 4.5 - 0.3 - 2.0 = 2.2
    if abs(gap - 2.2) > 1e-3:
        raise AssertionError(f"クリアランスがずれている: {gap}")
    print("geometry selftest ok")

if __name__ == "__main__":
    selftest()
