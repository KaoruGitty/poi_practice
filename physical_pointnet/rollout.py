"""1エピソードを進める。教師にも学習済み方針にも使う。"""

from __future__ import annotations

from typing import Optional, List, Tuple

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig
from .sim.env import NavEnv

@dataclass
class EpisodeResult:
    success: bool
    collision: bool
    stuck: bool
    steps: int
    final_dist: float
    world_name: str
    points: List[np.ndarray] = field(default_factory=list)
    status: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    trail: List[Tuple[float, float]] = field(default_factory=list)

def run_episode(
    env: NavEnv,
    act_fn,
    cfg: SimConfig,
    *,
    record: bool = False,
    noise_prob: float = 0.0,
    noise_width: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    on_step=None,
    execute_fn=None,
) -> EpisodeResult:
    """act_fn は記録する指令。execute_fn があるときは、実行だけそちらに替える。"""
    trail = [(env.state.x, env.state.y)]
    window_xy = (env.state.x, env.state.y)
    window_yaw = env.state.yaw
    points: List[np.ndarray] = []
    status: List[np.ndarray] = []
    action: List[np.ndarray] = []
    collision = False
    stuck = False
    success = False

    for t in range(cfg.max_steps):
        scan, state_vec = env.observe()
        v_cmd, w_cmd = act_fn(env)
        if record:
            points.append(scan.copy())
            status.append(state_vec.copy())
            action.append(np.array([v_cmd, w_cmd], dtype=np.float32))
        if on_step is not None:
            on_step(env, scan, trail)
        if execute_fn is None:
            v_apply, w_apply = v_cmd, w_cmd
        else:
            v_apply, w_apply = execute_fn(env)
        if (
            rng is not None
            and noise_prob > 0.0
            and rng.random() < noise_prob
            and env.goal_distance() > 1.3
        ):
            w_apply = float(np.clip(w_cmd + rng.uniform(-noise_width, noise_width), -cfg.w_max, cfg.w_max))
        collision = env.step(v_apply, w_apply)
        trail.append((env.state.x, env.state.y))
        if env.reached():
            success = True
            break
        if collision:
            break
        if t > 0 and (t + 1) % 25 == 0:
            moved = float(np.hypot(env.state.x - window_xy[0], env.state.y - window_xy[1]))
            turned = abs((env.state.yaw - window_yaw + np.pi) % (2 * np.pi) - np.pi)
            if moved < 0.2 and turned < 0.35 and env.goal_distance() > cfg.goal_radius:
                stuck = True
                break
            window_xy = (env.state.x, env.state.y)
            window_yaw = env.state.yaw

    return EpisodeResult(
        success=success,
        collision=collision,
        stuck=stuck,
        steps=env.steps,
        final_dist=env.goal_distance(),
        world_name=env.world.name,
        points=points,
        status=status,
        action=action,
        trail=trail,
    )
