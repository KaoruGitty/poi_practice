"""点群エンコーダ（PointNet++ か PTv3）と状態から、正規化済み速度指令を回帰する。"""

import argparse
import contextlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SimConfig
from .dataset import DemoDataset, collate_samples, load_demos, split_episodes
from .encoders import ENCODERS
from .evaluate import model_action, summarize
from .model import ModelConfig, build_model, save_checkpoint
from .preprocess import AugmentConfig, PreprocessConfig
from .rollout import run_episode
from .sim.env import NavEnv
from .sim.maps import all_maps

# Python 3.6 には contextlib.nullcontext がない。
try:
    nullcontext = contextlib.nullcontext
except AttributeError:
    @contextlib.contextmanager
    def nullcontext():
        yield


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def make_autocast(device, amp_dtype):
    """PyTorch 1.10 は torch.cuda.amp.autocast。dtype 引数は後発 API。"""
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    try:
        return torch.cuda.amp.autocast(dtype=amp_dtype)
    except TypeError:
        return torch.cuda.amp.autocast()


def make_grad_scaler(enabled):
    """PyTorch 1.10 は torch.cuda.amp.GradScaler のみ。"""
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, optimizer, device, omega_weight: float, amp_dtype, scaler) -> float:
    total = 0.0
    count = 0
    for points, status, action in loader:
        points = points.to(device)
        status = status.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        with make_autocast(device, amp_dtype):
            pred = model(points, status)
        err = pred.float() - action
        loss = (err[:, 0] ** 2).mean() + omega_weight * (err[:, 1] ** 2).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.item()) * status.shape[0]
        count += status.shape[0]
    return total / max(count, 1)


def closed_loop_probe(model, preprocess_cfg: PreprocessConfig, cfg: SimConfig, device, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    results = []
    for world in all_maps():
        env = NavEnv(world, cfg, rng)
        env.reset()
        results.append(run_episode(env, model_action(model, preprocess_cfg, cfg, device), cfg))
    model.train()
    return summarize(results)["success_rate"]


def train(args: argparse.Namespace) -> None:
    cfg = SimConfig()
    preprocess_cfg = PreprocessConfig.from_dict({**PreprocessConfig.for_sim(cfg).to_dict(), **load_json(args.preprocess_config)})
    augment = None if args.no_augment else AugmentConfig()
    model_cfg = ModelConfig(
        encoder=args.encoder,
        encoder_kwargs=load_json(args.encoder_config),
        in_channels=preprocess_cfg.in_channels,
        status_dim=cfg.status_dim,
        use_status=not args.no_status,
    )

    demos = load_demos(args.data)
    train_idx, val_idx = split_episodes(demos["episode_id"], args.val_ratio, args.seed)
    train_ds = DemoDataset(demos, train_idx, preprocess_cfg, augment)
    val_ds = DemoDataset(demos, val_idx, preprocess_cfg, None)
    device = pick_device(args.device)
    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=collate_samples,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    # persistent_workers は 1.7+。num_workers=0 のときは渡さない。
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CUDA 10.2 / torch 1.10 では AMP は float16 前提。
    amp_dtype = torch.float16 if (args.amp and device.type == "cuda") else None
    scaler = make_grad_scaler(enabled=amp_dtype is not None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    history = []
    best = float("inf")
    best_success = -1.0
    params = sum(p.numel() for p in model.parameters())
    print(f"装置={device}  エンコーダ={args.encoder}  状態入力={not args.no_status}  パラメータ={params:,}")
    print(f"学習サンプル={len(train_ds)}  検証サンプル={len(val_ds)}  入力チャネル={preprocess_cfg.in_channels}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, optimizer, device, args.omega_weight, amp_dtype, scaler)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device, args.omega_weight, amp_dtype, scaler)
        probe = None
        if args.probe_every and epoch % args.probe_every == 0:
            probe = closed_loop_probe(model, preprocess_cfg, cfg, device, seed=10000 + epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "success_rate": probe})
        message = f"epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}"
        if probe is not None:
            message += f"  閉ループ成功={probe:.0%}"
        print(message)

        def save(name: str) -> None:
            save_checkpoint(out / name, model, model_cfg, preprocess_cfg, cfg, epoch=epoch, val_loss=val_loss, args=vars(args))

        save("last.pt")
        if val_loss < best:
            best = val_loss
            save("best.pt")
        if probe is not None and probe >= best_success:
            best_success = probe
            save("best_closed.pt")

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=120)
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    ax.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最良の検証損失={best:.4f}  保存先={out / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="点群と状態から速度指令を学習する")
    parser.add_argument("--data", default="data/demos.npz")
    parser.add_argument("--out", default="", help="省略時は runs/<encoder>")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="pointnet2")
    parser.add_argument("--encoder-config", default="", help="エンコーダの設定を上書きする JSON")
    parser.add_argument("--preprocess-config", default="", help="前処理の設定を上書きする JSON")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--omega-weight", type=float, default=1.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-status", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="CUDA で混合精度（float16）を使う")
    args = parser.parse_args()
    args.out = args.out or f"runs/{args.encoder}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if pick_device(args.device).type == "cpu":
        torch.set_num_threads(1)
    train(args)


if __name__ == "__main__":
    main()
