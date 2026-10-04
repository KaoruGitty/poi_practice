"""室内ナビゲーション用の地図。低い障害物は水平スキャンでは見えない。"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import AABB


@dataclass(frozen=True)
class World:
    name: str
    title: str
    width: float
    height: float
    obstacles: tuple[AABB, ...]
    start_region: tuple[float, float, float, float]
    goal_region: tuple[float, float, float, float]

    def solids(self) -> list[AABB]:
        """レイキャスト用。室内の障害物に外壁を足す。"""
        thickness = 0.5
        wall_h = 2.2
        w = self.width
        h = self.height
        walls = [
            AABB(-thickness, w + thickness, -thickness, 0.0, 0.0, wall_h),
            AABB(-thickness, w + thickness, h, h + thickness, 0.0, wall_h),
            AABB(-thickness, 0.0, 0.0, h, 0.0, wall_h),
            AABB(w, w + thickness, 0.0, h, 0.0, wall_h),
        ]
        return walls + list(self.obstacles)


def _box(cx: float, cy: float, sx: float, sy: float, height: float) -> AABB:
    return AABB.from_center(cx, cy, sx, sy, height)


def room() -> World:
    return World(
        name="room",
        title="障害物のある部屋",
        width=14.0,
        height=10.0,
        obstacles=(
            _box(5.0, 3.0, 1.5, 1.1, 0.28),
            _box(7.2, 5.1, 0.8, 0.8, 1.8),
            _box(6.2, 8.1, 2.4, 0.55, 1.45),
            _box(9.6, 3.3, 1.2, 1.3, 0.9),
            _box(10.2, 7.4, 1.7, 0.7, 0.30),
        ),
        start_region=(0.9, 2.3, 1.0, 9.0),
        goal_region=(11.7, 13.1, 1.0, 9.0),
    )


def corridor() -> World:
    return World(
        name="corridor",
        title="クランク通路",
        width=18.0,
        height=8.0,
        obstacles=(
            _box(6.0, 2.55, 0.45, 5.1, 2.0),
            _box(12.0, 5.45, 0.45, 5.1, 2.0),
            _box(9.0, 4.0, 1.2, 0.8, 0.28),
        ),
        start_region=(0.9, 2.2, 1.2, 6.8),
        goal_region=(15.8, 17.1, 1.2, 6.8),
    )


def warehouse() -> World:
    return World(
        name="warehouse",
        title="倉庫の棚",
        width=16.0,
        height=12.0,
        obstacles=(
            _box(4.6, 2.5, 0.55, 3.6, 1.7),
            _box(4.6, 9.5, 0.55, 3.6, 1.7),
            _box(10.6, 2.5, 0.55, 3.6, 1.65),
            _box(10.6, 9.5, 0.55, 3.6, 1.65),
            _box(7.6, 9.2, 1.4, 0.8, 0.30),
            _box(13.2, 4.0, 1.1, 0.75, 0.32),
        ),
        start_region=(1.0, 2.6, 1.0, 3.6),
        goal_region=(13.4, 15.0, 8.6, 11.0),
    )


def slalom() -> World:
    return World(
        name="slalom",
        title="スラローム",
        width=16.0,
        height=9.0,
        obstacles=(
            _box(4.4, 2.7, 1.7, 4.2, 1.25),
            _box(8.4, 6.5, 1.6, 3.8, 0.28),
            _box(12.2, 2.6, 1.5, 4.0, 1.65),
        ),
        start_region=(0.9, 2.1, 3.4, 5.6),
        goal_region=(14.3, 15.2, 3.4, 5.6),
    )


def all_maps() -> list[World]:
    return [room(), corridor(), warehouse(), slalom()]
