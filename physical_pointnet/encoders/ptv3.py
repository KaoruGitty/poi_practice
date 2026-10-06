"""Point Transformer V3 エンコーダ（Wu ほか、CVPR 2024）。

構成は公式実装（Pointcept）のエンコーダ部分に合わせている。

1. 点をボクセル座標にし、同じボクセルの点を 1 つにまとめる。
2. 疎な 3D 畳み込みで埋め込む。
3. 各段で、空間充填曲線の順に点を並べ、patch_size 点ずつの区画で自己注意を掛ける。
   ブロックごとに並べ方（Z 曲線、ヒルベルト曲線と、その xy 入れ替え）を替える。
   位置の情報は、注意の前に置く疎な畳み込み（xCPE）で入れる。
4. 段の間でボクセルを stride 倍に粗くし、特徴を max でまとめる。
5. 最後の段の点をサンプルごとにまとめて 1 本のベクトルにする。

分割用のデコーダは持たない。疎な畳み込みは sparse_conv.SubMConv3d（PyTorch だけ）で、
注意は torch.nn.functional.scaled_dot_product_attention（CUDA では flash attention が使われる）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..points import PointBatch, global_pool, pooled_width, segment_reduce
from .base import PointEncoder
from .serialization import ORDERS, grid_depth, serialize
from .sparse_conv import SubMConv3d, neighbor_table


class _Stage:
    """1 段の中で共有する、ボクセル座標、直列化の並び、近傍表、区画の番号。"""

    def __init__(self, grid: torch.Tensor, batch: torch.Tensor, batch_size: int, orders: List[str]):
        self.grid = grid
        self.batch = batch
        self.batch_size = batch_size
        self.counts = torch.bincount(batch, minlength=batch_size)
        depth = grid_depth(grid)
        self.perms = [serialize(grid, batch, order, depth).argsort() for order in orders]
        self._tables: Dict[int, torch.Tensor] = {}
        self._patches: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, int]] = {}

    def table(self, kernel_size: int) -> torch.Tensor:
        if kernel_size not in self._tables:
            self._tables[kernel_size] = neighbor_table(self.grid, self.batch, kernel_size)
        return self._tables[kernel_size]

    def patches(self, order_index: int, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """区画に並べるための番号を返す。

        区画の大きさはバッチ内の最小点数を超えないようにする。各サンプルの点数を区画の
        倍数に足りるまで、直前の区画の点を複製して埋める（公式実装と同じ）。
        返り値は、区画に並べるときに取る元の点番号 (P*K,)、元の点ごとの区画上の位置 (M,)、K。
        """
        key = (order_index, patch_size)
        if key not in self._patches:
            counts = self.counts
            k = int(min(patch_size, int(counts.min())))
            padded = (counts + k - 1) // k * k
            start = torch.cumsum(counts, 0) - counts
            pad_start = torch.cumsum(padded, 0) - padded
            sample = torch.repeat_interleave(torch.arange(self.batch_size, device=counts.device), padded)
            local = torch.arange(int(padded.sum()), device=counts.device) - pad_start[sample]
            real = local < counts[sample]
            src = torch.where(real, local, local - k)
            gather = self.perms[order_index][start[sample] + src]
            back = torch.empty_like(gather[: self.grid.shape[0]])
            back[gather[real]] = torch.nonzero(real).squeeze(1)
            self._patches[key] = (gather, back, k)
        return self._patches[key]


class _DropPath(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x
        keep = 1.0 - self.prob
        return x * x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep


class _SerializedAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, patch_size: int, qkv_bias: bool, attn_drop: float, proj_drop: float):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels は num_heads で割り切れる数にしてください。")
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat: torch.Tensor, stage: _Stage, order_index: int) -> torch.Tensor:
        gather, back, k = stage.patches(order_index, self.patch_size)
        channels = feat.shape[1]
        qkv = self.qkv(feat[gather]).reshape(-1, k, 3, self.num_heads, channels // self.num_heads)
        q, key, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, key, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(-1, channels)[back]
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        cpe_kernel: int,
    ):
        super().__init__()
        self.cpe_kernel = cpe_kernel
        self.cpe_conv = SubMConv3d(channels, channels, cpe_kernel, bias=True)
        self.cpe_linear = nn.Linear(channels, channels)
        self.cpe_norm = nn.LayerNorm(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = _SerializedAttention(channels, num_heads, patch_size, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )
        self.drop_path = _DropPath(drop_path)

    def forward(self, feat: torch.Tensor, stage: _Stage, order_index: int) -> torch.Tensor:
        feat = feat + self.cpe_norm(self.cpe_linear(self.cpe_conv(feat, stage.table(self.cpe_kernel))))
        feat = feat + self.drop_path(self.attn(self.norm1(feat), stage, order_index))
        return feat + self.drop_path(self.mlp(self.norm2(feat)))


def _merge_voxels(
    grid: torch.Tensor, batch: torch.Tensor, coord: torch.Tensor, feat: torch.Tensor, reduce: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """同じサンプルの同じボクセルに入る点を 1 つにまとめる。結果はサンプル番号の昇順に並ぶ。"""
    size = int(grid.max()) + 1
    key = ((batch * size + grid[:, 0]) * size + grid[:, 1]) * size + grid[:, 2]
    uniq, inverse = torch.unique(key, return_inverse=True)
    n = uniq.numel()
    new_batch = uniq // (size**3)
    rest = uniq % (size**3)
    new_grid = torch.stack([rest // (size * size), (rest // size) % size, rest % size], dim=1)
    new_coord = segment_reduce(coord, inverse, n, "mean")
    new_feat = segment_reduce(feat, inverse, n, reduce)
    return new_grid, new_batch, new_coord, new_feat


class _GridPooling(nn.Module):
    """ボクセルを stride 倍に粗くし、全結合のあと max でまとめる。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, grid, batch, coord, feat):
        grid, batch, coord, feat = _merge_voxels(grid // self.stride, batch, coord, self.proj(feat), "max")
        return grid, batch, coord, self.act(self.norm(feat))


class PTv3Encoder(PointEncoder):
    """Point Transformer V3 のエンコーダ。既定値はこのリポジトリのシミュレータ向けの小さな構成。"""

    def __init__(
        self,
        in_channels: int,
        grid_size: float = 0.1,
        orders: Tuple[str, ...] = ORDERS,
        shuffle_orders: bool = True,
        stride: Tuple[int, ...] = (2, 2, 2),
        enc_depths: Tuple[int, ...] = (1, 1, 1, 1),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16),
        enc_patch_size: Tuple[int, ...] = (64, 64, 64, 64),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.1,
        embed_kernel: int = 5,
        cpe_kernel: int = 3,
        pool: str = "max",
    ):
        super().__init__()
        stages = len(enc_depths)
        if not (len(enc_channels) == len(enc_num_head) == len(enc_patch_size) == stages == len(stride) + 1):
            raise ValueError("enc_* は段の数、stride は段の数より 1 少ない長さにしてください。")
        for order in orders:
            if order not in ORDERS:
                raise ValueError(f"order は {', '.join(ORDERS)} のどれかです。")
        self.grid_size = grid_size
        self.orders = list(orders)
        self.shuffle_orders = shuffle_orders
        self.pool = pool
        self.embed_kernel = embed_kernel
        self.embed_conv = SubMConv3d(in_channels, enc_channels[0], embed_kernel)
        self.embed_norm = nn.BatchNorm1d(enc_channels[0])
        self.embed_act = nn.GELU()
        rates = torch.linspace(0.0, drop_path, sum(enc_depths)).tolist()
        self.pools = nn.ModuleList(
            _GridPooling(enc_channels[s - 1], enc_channels[s], stride[s - 1]) for s in range(1, stages)
        )
        self.stages = nn.ModuleList()
        used = 0
        for s in range(stages):
            blocks = nn.ModuleList(
                _Block(
                    enc_channels[s],
                    enc_num_head[s],
                    enc_patch_size[s],
                    mlp_ratio,
                    qkv_bias,
                    attn_drop,
                    proj_drop,
                    rates[used + i],
                    cpe_kernel,
                )
                for i in range(enc_depths[s])
            )
            used += enc_depths[s]
            self.stages.append(blocks)
        self.out_channels = pooled_width(enc_channels[-1], pool)

    def _stage_orders(self) -> List[str]:
        if self.training and self.shuffle_orders:
            return [self.orders[i] for i in torch.randperm(len(self.orders)).tolist()]
        return list(self.orders)

    def forward(self, points: PointBatch) -> torch.Tensor:
        coord, feat, batch = points.coord, points.feat, points.batch
        origin = segment_reduce(coord, batch, points.batch_size, "min")
        grid = torch.floor((coord - origin[batch]) / self.grid_size).long()
        grid, batch, coord, feat = _merge_voxels(grid, batch, coord, feat, "mean")

        stage = _Stage(grid, batch, points.batch_size, self._stage_orders())
        feat = self.embed_act(self.embed_norm(self.embed_conv(feat, stage.table(self.embed_kernel))))
        for s, blocks in enumerate(self.stages):
            if s > 0:
                grid, batch, coord, feat = self.pools[s - 1](grid, batch, coord, feat)
                stage = _Stage(grid, batch, points.batch_size, self._stage_orders())
            for i, block in enumerate(blocks):
                feat = block(feat, stage, i % len(self.orders))
        return global_pool(feat, batch, points.batch_size, self.pool)
