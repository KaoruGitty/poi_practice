"""古典プランナを走らせ、成功したエピソードだけを保存する。"""

from typing import List

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SimConfig
from .dataset import save_demos
from .obs import normalize_action
from .rollout import run_episode
from .sim.env import NavEnv
from .sim.expert import expert_action
from .sim.maps import all_maps


def collect(
    episodes: int,
    seed: int,
    out: str,
    max_attempts_factor: int = 8,
    ckpt: str = "",
    append: str = "",
) -> dict:
    cfg = SimConfig()
    maps = all_maps()
    rng = np.random.default_rng(seed)
    policy = None
    if ckpt:
        from .evaluate import load_policy

        policy, cfg = load_policy(ckpt)
    points: List[np.ndarray] = []
    status: List[np.ndarray] = []
    action: List[np.ndarray] = []
    episode_id: List[np.ndarray] = []
    map_id: List[np.ndarray] = []
    kept = 0
    successes = 0
    attempts = 0
    failures = {"collision": 0, "stuck": 0, "timeout": 0}
    limit = episodes * max_attempts_factor

    while kept < episodes and attempts < limit:
        map_index = attempts % len(maps)
        world = maps[map_index]
        env = NavEnv(world, cfg, rng)
        env.reset()

        def label(env, world=world):
            return expert_action(env.state, env.goal, world, cfg)

        execute = None
        if policy is not None:
            def execute(env, world=world, label=label):
                if rng.random() < 0.75:
                    return policy(env)
                return label(env)

        result = run_episode(
            env,
            label,
            cfg,
            record=True,
            noise_prob=0.0 if policy is not None else 0.18,
            noise_width=0.35,
            rng=rng,
            execute_fn=execute,
        )
        attempts += 1
        useful = len(result.action) >= 12 and (result.success or policy is not None)
        if not useful:
            if result.collision:
                failures["collision"] += 1
            elif result.stuck:
                failures["stuck"] += 1
            else:
                failures["timeout"] += 1
            print(
                f"破棄 {attempts:03d}  {world.title}  "
                f"衝突={int(result.collision)} 停滞={int(result.stuck)}  残距離={result.final_dist:.2f} m"
            )
            continue
        n = len(result.action)
        points.append(np.stack(result.points))
        status.append(np.stack(result.status))
        action.append(np.stack([normalize_action(float(a[0]), float(a[1]), cfg) for a in result.action]))
        episode_id.append(np.full(n, kept, dtype=np.int32))
        map_id.append(np.full(n, map_index, dtype=np.int32))
        kept += 1
        successes += int(result.success)
        print(f"採用 {kept:02d}/{episodes}  {world.title}  歩数={n}  成功={int(result.success)}  試行={attempts}")

    if kept == 0:
        raise RuntimeError("成功エピソードを集められませんでした。教師のパラメータを見直してください。")

    payload = {
        "points": np.concatenate(points).astype(np.float32),
        "status": np.concatenate(status).astype(np.float32),
        "action": np.concatenate(action).astype(np.float32),
        "episode_id": np.concatenate(episode_id),
        "map_id": np.concatenate(map_id),
    }
    if append:
        from .dataset import load_demos

        old = load_demos(append)
        shift = int(old["episode_id"].max()) + 1
        payload = {
            "points": np.concatenate([old["points"], payload["points"]]),
            "status": np.concatenate([old["status"], payload["status"]]),
            "action": np.concatenate([old["action"], payload["action"]]),
            "episode_id": np.concatenate([old["episode_id"], payload["episode_id"] + shift]),
            "map_id": np.concatenate([old["map_id"], payload["map_id"]]),
        }
    save_demos(out, payload, cfg)
    summary = {
        "episodes": kept,
        "attempts": attempts,
        "samples": int(payload["points"].shape[0]),
        "points_per_scan": int(payload["points"].shape[1]),
        "failures": failures,
        "out": out,
    }
    stats_path = Path(out).with_suffix(".json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"保存 {out}  サンプル={summary['samples']}  "
        f"採用中の到達={successes / kept:.0%}  破棄={failures}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDARと身体状態の教師デモを集める")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/demos.npz")
    parser.add_argument("--ckpt", default="", help="指定すると、この方針を混ぜて走らせ、ラベルは教師のまま集める")
    parser.add_argument("--append", default="", help="既存の npz に結合する")
    args = parser.parse_args()
    collect(args.episodes, args.seed, args.out, ckpt=args.ckpt, append=args.append)


if __name__ == "__main__":
    main()
