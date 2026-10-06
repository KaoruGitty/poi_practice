"""docs/images/ の図を作る。シミュレータのスキャンを実際のエンコーダの関数に通して描く。

    MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/make_doc_figures.py

学習済みの重みは使わない。重みが関わる図（max の様子）は乱数で初期化したネットワークの値。
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_pointnet.config import SimConfig  # noqa: E402
from physical_pointnet.encoders.pointnet2 import PointNet2Encoder  # noqa: E402
from physical_pointnet.encoders.ptv3 import _merge_voxels, _Stage  # noqa: E402
from physical_pointnet.encoders.serialization import (  # noqa: E402
    ORDERS,
    grid_depth,
    hilbert_encode,
    serialize,
    z_order_encode,
)
from physical_pointnet.encoders.sparse_conv import neighbor_table  # noqa: E402
from physical_pointnet.points import (  # noqa: E402
    PointBatch,
    ball_query,
    farthest_point_sample,
    gather_points,
    segment_reduce,
    to_padded,
)
from physical_pointnet.preprocess import PreprocessConfig, preprocess_scan  # noqa: E402
from physical_pointnet.sim.env import NavEnv  # noqa: E402
from physical_pointnet.sim.maps import all_maps  # noqa: E402

OUT = ROOT / "docs" / "images"
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def scans() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    cfg = SimConfig()
    pre = PreprocessConfig.for_sim(cfg)
    coords, feats = [], []
    for i, world in enumerate(all_maps()[:3]):
        env = NavEnv(world, cfg, np.random.default_rng(i))
        env.reset()
        points, _ = env.observe()
        c, f, _ = preprocess_scan(points, pre)
        coords.append(c)
        feats.append(f)
    return coords, feats

def top_view(ax, title: Optional[str] = None, lim: float = 8.0) -> None:
    """上から見た図。縦が前（x）、横が左右（y、左が正なので軸を反転）。"""
    ax.set_aspect("equal")
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("y [m]（左が正）")
    ax.set_ylabel("x [m]（前が正）")
    ax.plot(0, 0, marker="^", color="k", markersize=9)
    if title:
        ax.set_title(title)

def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ---------------------------------------------------------------- PointNet++

def pn2_figures(coords: List[np.ndarray], feats: List[np.ndarray]) -> None:
    pb = PointBatch.from_arrays(coords, feats)
    c0 = coords[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc = axes[0].scatter(c0[:, 1], c0[:, 0], c=c0[:, 2], s=8, cmap="viridis")
    top_view(axes[0], f"上から見たスキャン（{len(c0)} 点、色は高さ）")
    fig.colorbar(sc, ax=axes[0], label="z [m]")
    r = np.linalg.norm(c0[:, :2], axis=1)
    axes[1].scatter(r, c0[:, 2], s=8, c=c0[:, 2], cmap="viridis")
    axes[1].set_xlabel("ロボットからの水平距離 [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("横から見た分布（仰角 4 本のリング）")
    save(fig, "pn2_scan.png")

    xyz, mask = to_padded(pb.coord, pb.batch, pb.batch_size)
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.imshow(mask.numpy(), aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1.4)
    ax.set_yticks(range(pb.batch_size), [f"サンプル{b}（{int(n)} 点）" for b, n in enumerate(pb.counts)])
    ax.set_xlabel("表の列（点の番号）")
    ax.set_title(f"to_padded の mask：黒が本物の点、白が詰め物　形 {tuple(mask.shape)}")
    save(fig, "pn2_padded.png")

    one = torch.from_numpy(c0).unsqueeze(0)
    one_mask = torch.ones(1, len(c0), dtype=torch.bool)
    order = farthest_point_sample(one, one_mask, 64)[0].numpy()
    rand = np.random.default_rng(0).choice(len(c0), 64, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, n in zip(axes[:3], (4, 16, 64)):
        ax.scatter(c0[:, 1], c0[:, 0], s=4, color="0.75")
        sel = c0[order[:n]]
        ax.scatter(sel[:, 1], sel[:, 0], s=30, color="tab:red")
        for i in range(min(n, 6)):
            ax.annotate(str(i + 1), (c0[order[i], 1], c0[order[i], 0]), fontsize=9, xytext=(4, 4), textcoords="offset points")
        top_view(ax, f"最遠点サンプリング 最初の {n} 点")
    axes[3].scatter(c0[:, 1], c0[:, 0], s=4, color="0.75")
    axes[3].scatter(c0[rand, 1], c0[rand, 0], s=30, color="tab:blue")
    top_view(axes[3], "比較：ランダムに 64 点")
    save(fig, "pn2_fps.png")

    centers = one[0, order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    in_view = np.where(np.abs(centers.numpy()[:, :2]).max(axis=1) < 4.5)[0]
    picks = [int(in_view[0]), int(in_view[len(in_view) // 2]), int(in_view[-1])]
    colors = ["tab:red", "tab:green", "tab:purple"]
    for ax, radius, k in zip(axes, (0.8, 1.6), (8, 16)):
        idx, valid = ball_query(centers.unsqueeze(0), torch.ones(1, 64, dtype=torch.bool), one, one_mask, radius, k)
        ax.scatter(c0[:, 1], c0[:, 0], s=4, color="0.8")
        for p, col in zip(picks, colors):
            cx, cy = centers[p, 1].item(), centers[p, 0].item()
            ax.add_patch(Circle((cx, cy), radius, fill=False, color=col, linestyle="--"))
            nb = c0[idx[0, p].numpy()]
            ok = valid[0, p].numpy()
            ax.scatter(nb[ok, 1], nb[ok, 0], s=22, color=col)
            ax.scatter(nb[~ok, 1], nb[~ok, 0], s=40, color=col, marker="x")
            ax.plot(cx, cy, marker="*", color="k", markersize=12)
        top_view(ax, f"半径 {radius} m・最大 {k} 点　● 有効　× 近い順 {k} 個に入ったが半径の外", lim=5.5)
    save(fig, "pn2_ball_query.png")

    enc = PointNet2Encoder(3).eval()
    layer = enc.layers[0]
    idx, valid = ball_query(centers.unsqueeze(0), torch.ones(1, 64, dtype=torch.bool), one, one_mask, 0.8, 8)
    n_valid = valid[0].sum(-1).numpy()
    p = int(np.where((n_valid >= 4) & (n_valid <= 6))[0][0])
    nb_idx, ok = idx[0, p], valid[0, p]
    center = centers[p]
    absolute = one[0, nb_idx]
    rel = (absolute - center) / 0.8
    print(f"pn2 group: center {p} at {center.numpy().round(2)} valid {int(ok.sum())}/8")
    print("  neighbor idx  ", nb_idx.tolist())
    print("  valid         ", ok.int().tolist())
    print("  dist [m]      ", (absolute - center).norm(dim=1).numpy().round(2).tolist())
    print("  rel/radius    ", rel.numpy().round(2).tolist())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    zero = torch.zeros(3)
    panels = [
        (axes[0], rel, (1, 0), zero, 1.0, "上から（x と y）", ("相対 y ÷ 半径", "相対 x ÷ 半径")),
        (axes[1], rel, (1, 2), zero, 1.0, "正面から（y と z）", ("相対 y ÷ 半径", "相対 z ÷ 半径")),
        (axes[2], rel, (0, 2), zero, 1.0, "横から（x と z）", ("相対 x ÷ 半径", "相対 z ÷ 半径")),
    ]
    for ax, pts, (h, v), c, r, title, (xl, yl) in panels:
        ax.scatter(pts[ok, h], pts[ok, v], color="tab:green", s=45, label="有効", zorder=3)
        ax.scatter(pts[~ok, h], pts[~ok, v], color="tab:red", marker="x", s=60, label="半径の外", zorder=3)
        for j in range(8):
            ax.annotate(str(j), (pts[j, h].item(), pts[j, v].item()), fontsize=9, xytext=(5, 3), textcoords="offset points")
        ax.plot(c[h], c[v], "k*", markersize=15, label="中心")
        ax.add_patch(Circle((c[h].item(), c[v].item()), r, fill=False, ls="--"))
        ax.set_aspect("equal")
        lim = 3.6 * r
        if h == 1:
            ax.set_xlim(c[h].item() + lim, c[h].item() - lim)
        else:
            ax.set_xlim(c[h].item() - lim, c[h].item() + lim)
        ax.set_ylim(c[v].item() - lim, c[v].item() + lim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"中心 {p} の近傍 8 点を (点 − 中心) ÷ 半径 0.8 m にしたもの。破線が半径（= 1）の球の断面")
    save(fig, "pn2_relative.png")

    with torch.no_grad():
        grouped = torch.cat([rel, torch.from_numpy(feats[0])[nb_idx]], dim=-1)
        hidden = torch.full((8, layer.branches[0].out_channels), float("-inf"))
        hidden[ok] = layer.branches[0](grouped[ok])
    shown = hidden[:, :16].numpy()
    pooled = shown.max(axis=0, keepdims=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={"height_ratios": [8, 1.3]})
    data = np.where(np.isfinite(shown), shown, np.nan)
    vmax = float(np.nanmax(data))
    axes[0].imshow(np.ma.masked_invalid(data), aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    for r in np.where(~ok.numpy())[0]:
        axes[0].add_patch(Rectangle((-0.5, r - 0.5), 16, 1, color="0.85"))
        axes[0].text(7.5, r, "無効（-inf）", ha="center", va="center", fontsize=9)
    axes[0].set_yticks(range(8), [f"近傍 {i}" for i in range(8)])
    axes[0].set_xticks([])
    axes[0].set_title("全結合の出力（近傍 8 点 × 先頭 16 チャネル、乱数初期化の重み）")
    print("  max of first 8 channels", pooled[0, :8].round(2).tolist())
    axes[1].imshow(pooled, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_yticks([0], ["max"])
    axes[1].set_xlabel("チャネル")
    save(fig, "pn2_maxpool.png")

    with torch.no_grad():
        s_xyz, s_feat, s_mask = to_padded(pb.coord, pb.batch, pb.batch_size)[0], to_padded(pb.feat, pb.batch, pb.batch_size)[0], mask
        stages = [(s_xyz[0, : len(c0)].numpy(), s_feat.shape[-1])]
        for sa in enc.layers:
            s_xyz, s_feat, s_mask = sa(s_xyz, s_feat, s_mask, False)
            stages.append((s_xyz[0][s_mask[0]].numpy(), s_feat.shape[-1]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    titles = ["入力", "段 1 の出力", "段 2 の出力"]
    for ax, (pts, ch), title in zip(axes, stages, titles):
        ax.scatter(c0[:, 1], c0[:, 0], s=3, color="0.85")
        ax.scatter(pts[:, 1], pts[:, 0], s=12 if len(pts) > 100 else 40, color="tab:red")
        top_view(ax, f"{title}：{len(pts)} 点 × {ch} 次元")
    save(fig, "pn2_hierarchy.png")

# ---------------------------------------------------------------- PTv3

def ptv3_figures(coords: List[np.ndarray], feats: List[np.ndarray]) -> None:
    pb = PointBatch.from_arrays(coords[:2], feats[:2])
    grid_size = 0.1
    origin = segment_reduce(pb.coord, pb.batch, pb.batch_size, "min")
    grid = torch.floor((pb.coord - origin[pb.batch]) / grid_size).long()
    grid, batch, coord, feat = _merge_voxels(grid, pb.batch, pb.coord, pb.feat, "mean")
    sel0 = batch == 0
    c0 = coord[sel0].numpy()
    g0 = grid[sel0].numpy()
    o0 = origin[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    axes[0].scatter(c0[:, 1], c0[:, 0], s=4, color="0.4")
    box = (1.0, 3.6, -2.6, 0.0)  # y_min, y_max, x_min, x_max（拡大する範囲）
    axes[0].add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, color="tab:red", lw=2))
    top_view(axes[0], "スキャン全体（赤枠を右で拡大）")
    ax = axes[1]
    inside = (c0[:, 1] >= box[0]) & (c0[:, 1] <= box[1]) & (c0[:, 0] >= box[2]) & (c0[:, 0] <= box[3])
    for gx, gy in {(int(g[0]), int(g[1])) for g in g0[inside]}:
        x0 = o0[0] + gx * grid_size
        y0 = o0[1] + gy * grid_size
        ax.add_patch(Rectangle((y0, x0), grid_size, grid_size, color="tab:orange", alpha=0.5))
    for v in np.arange(np.floor((box[0] - o0[1]) / grid_size), np.ceil((box[1] - o0[1]) / grid_size) + 1):
        ax.axvline(o0[1] + v * grid_size, color="0.85", lw=0.5)
    for v in np.arange(np.floor((box[2] - o0[0]) / grid_size), np.ceil((box[3] - o0[0]) / grid_size) + 1):
        ax.axhline(o0[0] + v * grid_size, color="0.85", lw=0.5)
    ax.scatter(c0[inside, 1], c0[inside, 0], s=14, color="k", zorder=3)
    ax.set_xlim(box[1], box[0])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"格子 {grid_size} m：点のある格子（橙）だけが残る")
    save(fig, "ptv3_voxel.png")

    cells = torch.tensor([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])
    fig = plt.figure(figsize=(12, 5.6))
    for i, (name, fn) in enumerate((("Z 曲線（z）", z_order_encode), ("ヒルベルト曲線（hilbert）", hilbert_encode))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        path = cells[fn(cells, 2).argsort()].numpy()
        for j in range(len(path) - 1):
            ax.plot(*path[j : j + 2].T, color=plt.cm.viridis(j / len(path)), lw=2)
        ax.scatter(*path[0], color="k", s=40)
        jumps = np.abs(np.diff(path, axis=0)).sum(1)
        ax.set_title(f"{name}　4×4×4\n隣の番号どうしの距離 最大 {int(jumps.max())}、面で接していない継ぎ目 {int((jumps > 1).sum())} か所")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    save(fig, "ptv3_curves.png")

    depth = grid_depth(grid)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    rng = np.random.default_rng(0)
    orders = {"ランダムな順": rng.permutation(len(c0))}
    for name in ("z", "hilbert"):
        orders[name] = serialize(grid[sel0], batch[sel0], name, depth).argsort().numpy()
    for ax, (name, perm) in zip(axes, orders.items()):
        path = c0[perm]
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        for j in range(len(path) - 1):
            ax.plot(path[j : j + 2, 1], path[j : j + 2, 0], color=plt.cm.viridis(j / len(path)), lw=1)
        ax.scatter(path[:, 1], path[:, 0], s=5, color="k", zorder=3)
        top_view(ax, f"{name}：隣どうしの距離 中央値 {np.median(step):.2f} m・平均 {step.mean():.2f} m")
    save(fig, "ptv3_serialized_scan.png")

    stage = _Stage(grid, batch, pb.batch_size, list(ORDERS))
    k_patch = 64
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    cmap = plt.get_cmap("tab10")
    for i, (ax, name) in enumerate(zip(axes, ORDERS)):
        gather, back, k = stage.patches(i, k_patch)
        patch_id = (back[sel0] // k).numpy()
        for pid in np.unique(patch_id):
            m = patch_id == pid
            ax.scatter(c0[m, 1], c0[m, 0], s=9, color=cmap(int(pid) % 10), label=f"区画 {pid}")
        top_view(ax, f"{name}（K = {k}）")
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, "ptv3_patches.png")

    gather, back, k = stage.patches(2, k_patch)
    counts = torch.bincount(batch).tolist()
    padded = [(n + k - 1) // k * k for n in counts]
    slot = np.arange(gather.numel())
    real = back[gather].numpy() == slot
    fig, ax = plt.subplots(figsize=(14, 2.6))
    start = 0
    for b, n in enumerate(padded):
        for s in range(n):
            pos = start + s
            color = cmap((s // k) % 10) if real[pos] else "0.85"
            ax.add_patch(Rectangle((s, b), 1, 0.8, color=color, lw=0))
        for edge in range(0, n + 1, k):
            ax.axvline(edge, ymin=0, ymax=1, color="k", lw=0.4)
        ax.text(n + 3, b + 0.4, f"本物 {counts[b]} 点 + 複製 {n - counts[b]} 点 = {n}（{n // k} 区画）", va="center", fontsize=9)
        start += n
    ax.set_xlim(0, max(padded) + 120)
    ax.set_ylim(-0.2, len(padded))
    ax.set_yticks([0.4, 1.4], ["サンプル0", "サンプル1"])
    ax.invert_yaxis()
    ax.set_xlabel("区画に並べたときの位置（色は区画、灰色は直前の区画からの複製）")
    ax.set_title(f"hilbert の順で K = {k} 点ずつ区切った並び（gather の形 {tuple(gather.shape)}）")
    save(fig, "ptv3_patch_layout.png")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    g, b, c, f = grid, batch, coord, feat
    size = grid_size
    stats = []
    for s, ax in enumerate(axes):
        if s > 0:
            g, b, c, f = _merge_voxels(g // 2, b, c, f, "mean")
            size *= 2
        m = b == 0
        ax.scatter(c0[:, 1], c0[:, 0], s=2, color="0.75")
        for gx, gy in {(int(v[0]), int(v[1])) for v in g[m].numpy()}:
            ax.add_patch(Rectangle((o0[1] + gy * size, o0[0] + gx * size), size, size, color="tab:blue", alpha=0.35, lw=0))
        table = neighbor_table(g, b, 3)
        found = float((table < len(g)).sum(1).float().mean())
        stats.append((s + 1, size, int(m.sum()), found))
        top_view(ax, f"段 {s + 1}：格子 {size:.1f} m・{int(m.sum())} 点\n3×3×3 の近傍に平均 {found:.1f} 点")
    save(fig, "ptv3_pooling.png")
    for row in stats:
        print("stage %d grid %.1f m points %d mean neighbors(3^3) %.2f" % row)

    g3, b3 = grid, batch
    for _ in range(2):
        g3, b3, _, _ = _merge_voxels(g3 // 2, b3, torch.zeros(len(g3), 3), torch.zeros(len(g3), 1), "mean")
    table = neighbor_table(g3, b3, 3)
    found = (table < len(g3)).sum(1)
    p = int(found.argmax())
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 2.4]})
    row = table[p].numpy().reshape(3, 3, 3)
    for dz, ax in enumerate(axes[:3]):
        cell = row[:, :, dz]
        ax.imshow(cell < len(g3), cmap="Greens", vmin=0, vmax=1.3)
        for (i, j), v in np.ndenumerate(cell):
            ax.text(j, i, "なし" if v == len(g3) else str(v), ha="center", va="center", fontsize=9)
        ax.set_title(f"dz = {dz - 1}")
        ax.set_xticks(range(3), ["dy-1", "dy0", "dy+1"])
        ax.set_yticks(range(3), ["dx-1", "dx0", "dx+1"])
    axes[3].imshow((table[:40] < len(g3)).numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=1.3, interpolation="nearest")
    axes[3].set_xlabel("カーネルの位置（27 通り）")
    axes[3].set_ylabel("点の番号（先頭 40）")
    axes[3].set_title(f"近傍表 (M={len(g3)}, 27) の一部：緑が近傍あり")
    fig.suptitle(f"段 3（格子 0.4 m）の点 {p} の近傍表の行（数字は近傍の点番号、なしは 0 の行を指す）")
    save(fig, "ptv3_neighbor_table.png")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    coords, feats = scans()
    pn2_figures(coords, feats)
    ptv3_figures(coords, feats)

if __name__ == "__main__":
    main()
