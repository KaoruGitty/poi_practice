"""上面図のフレーム。LiDARは高さで色を分ける。"""

from typing import List, Tuple

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from ..config import SimConfig
from .geometry import to_world
from .maps import World
from .robot import RobotState

_FONT_READY = False


def _use_japanese_font() -> None:
    global _FONT_READY
    if _FONT_READY:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "AppleGothic", "Arial Unicode MS"):
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def _height_color(height: float) -> Tuple[float, float, float]:
    mix = float(np.clip((height - 0.25) / 1.55, 0.0, 1.0))
    low = np.array([0.42, 0.66, 0.82])
    high = np.array([0.93, 0.70, 0.34])
    return tuple(low * (1.0 - mix) + high * mix)


def render_frame(
    world: World,
    state: RobotState,
    goal: np.ndarray,
    points_body: np.ndarray,
    trail: List[Tuple[float, float]],
    cfg: SimConfig,
    title: str,
) -> Image.Image:
    _use_japanese_font()
    fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=110)
    fig.patch.set_facecolor("#121418")
    ax.set_facecolor("#1b1f27")
    ax.add_patch(
        plt.Rectangle((0.0, 0.0), world.width, world.height, facecolor="#232833", edgecolor="#8d97a8", linewidth=1.2)
    )
    for box in world.obstacles:
        ax.add_patch(
            plt.Rectangle(
                (box.xmin, box.ymin),
                box.xmax - box.xmin,
                box.ymax - box.ymin,
                facecolor=_height_color(box.zmax),
                edgecolor="#101218",
                linewidth=0.6,
                zorder=2,
            )
        )
    world_points = to_world(points_body, state.x, state.y, state.yaw)
    ground = world_points[:, 2] < 0.08
    hits = ~ground
    if np.any(ground):
        ax.scatter(
            world_points[ground, 0],
            world_points[ground, 1],
            s=3,
            c="#5c6b7a",
            alpha=0.25,
            linewidths=0,
            zorder=3,
        )
    if np.any(hits):
        colors = world_points[hits, 2]
        ax.scatter(
            world_points[hits, 0],
            world_points[hits, 1],
            s=8,
            c=colors,
            cmap="cool",
            vmin=0.0,
            vmax=1.8,
            alpha=0.9,
            linewidths=0,
            zorder=4,
        )
    if len(trail) > 1:
        xs, ys = zip(*trail)
        ax.plot(xs, ys, color="#ffb020", linewidth=1.6, alpha=0.95, zorder=5)
    ax.scatter([goal[0]], [goal[1]], s=160, facecolors="none", edgecolors="#ff5d5d", linewidths=2.0, zorder=6)
    ax.scatter([goal[0]], [goal[1]], s=16, c="#ff5d5d", zorder=6)

    nose = cfg.robot_radius * 1.55
    side = cfg.robot_radius * 0.95
    back = cfg.robot_radius * 0.85
    local = np.array([[nose, 0.0], [-back, side], [-back, -side]])
    c = math.cos(state.yaw)
    s = math.sin(state.yaw)
    rotated = np.column_stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]])
    rotated += np.array([state.x, state.y])
    ax.add_patch(plt.Polygon(rotated, closed=True, facecolor="#f4f6f8", edgecolor="#111318", zorder=7))

    dist = math.hypot(goal[0] - state.x, goal[1] - state.y)
    ax.text(
        0.02,
        0.98,
        f"{title}\nv {state.v:.2f} m/s    ω {state.w:+.2f} rad/s\n目標まで {dist:.2f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#f2f4f8",
        fontsize=10,
        zorder=8,
    )
    ax.set_xlim(-0.4, world.width + 0.4)
    ax.set_ylim(-0.4, world.height + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba[..., :3].copy())
    plt.close(fig)
    return image


def save_gif(frames: List[Image.Image], path: str, duration_ms: int) -> None:
    if not frames:
        raise RuntimeError("フレームが空です")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
