"""学習済み方針、または教師をシミュレータで走らせる。"""

from typing import Optional, List, Tuple

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import SimConfig
from .model import load_checkpoint
from .obs import denormalize_action
from .points import PointBatch
from .preprocess import PreprocessConfig, preprocess_scan
from .rollout import EpisodeResult, run_episode
from .sim.env import NavEnv
from .sim.expert import expert_action
from .sim.maps import all_maps
from .sim.render import render_frame, save_gif

def summarize(results: List[EpisodeResult]) -> dict:
    n = max(len(results), 1)
    return {
        "episodes": len(results),
        "success_rate": sum(r.success for r in results) / n,
        "collision_rate": sum(r.collision for r in results) / n,
        "stuck_rate": sum(r.stuck for r in results) / n,
        "mean_steps": float(np.mean([r.steps for r in results])) if results else 0.0,
        "mean_final_dist": float(np.mean([r.final_dist for r in results])) if results else 0.0,
    }

def model_action(model, preprocess_cfg: PreprocessConfig, cfg: SimConfig, device: torch.device):
    def act(env: NavEnv) -> Tuple[float, float]:
        points, status = env.observe()
        coord, feat, _ = preprocess_scan(points, preprocess_cfg)
        batch = PointBatch.from_arrays([coord], [feat]).to(device)
        with torch.no_grad():
            pred = model(batch, torch.from_numpy(status).float().unsqueeze(0).to(device))
        return denormalize_action(pred.squeeze(0).float().cpu().numpy(), cfg)

    return act

def load_policy(ckpt_path: str, device: Optional[torch.device] = None):
    model, _, preprocess_cfg, cfg, _ = load_checkpoint(ckpt_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model_action(model, preprocess_cfg, cfg, device), cfg

def evaluate(args: argparse.Namespace) -> dict:
    maps = all_maps()
    if args.expert:
        cfg = SimConfig()
        act = lambda env: expert_action(env.state, env.goal, env.world, cfg)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt か --expert を指定してください")
        act, cfg = load_policy(args.ckpt)
    rng = np.random.default_rng(args.seed)
    results = []
    gif_frames = []
    stride = 2
    for i in range(args.episodes):
        world = maps[i % len(maps)]
        env = NavEnv(world, cfg, rng)
        env.reset()

        def on_step(env, scan, trail, world=world, record_gif=(i == 0 and args.gif)):
            if record_gif and env.steps % stride == 0:
                gif_frames.append(
                    render_frame(world, env.state, env.goal, scan, list(trail), cfg, world.title)
                )

        result = run_episode(env, act, cfg, on_step=on_step if args.gif and i == 0 else None)
        results.append(result)
        print(
            f"{i + 1:02d} {world.title}  成功={int(result.success)}  "
            f"衝突={int(result.collision)}  停滞={int(result.stuck)}  "
            f"歩数={result.steps}  残距離={result.final_dist:.2f} m"
        )
    summary = summarize(results)
    print(
        f"成功率={summary['success_rate']:.0%}  衝突率={summary['collision_rate']:.0%}  "
        f"平均歩数={summary['mean_steps']:.0f}  平均残距離={summary['mean_final_dist']:.2f} m"
    )
    if args.gif:
        if not gif_frames:
            raise RuntimeError("GIF用のフレームがありません")
        path = Path(args.gif)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(gif_frames, str(path), duration_ms=int(cfg.dt * stride * 1000))
        gif_frames[len(gif_frames) // 2].save(path.with_name(path.stem + "_frame.png"))
        summary["gif"] = str(path)
        print(f"動画 {path}")
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="シミュレータ上でロボットを走らせて評価する")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gif", default="")
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
