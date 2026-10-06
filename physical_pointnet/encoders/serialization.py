"""PTv3 の直列化。ボクセル座標を空間充填曲線の番号に変え、近い点が近い順番に並ぶようにする。"""

import torch

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def z_order_encode(grid: torch.Tensor, depth: int) -> torch.Tensor:
    """(M, 3) の非負整数座標を Z 曲線（モートン順）の番号にする。"""
    code = torch.zeros(grid.shape[0], dtype=torch.long, device=grid.device)
    for bit in range(depth):
        for axis in range(3):
            code |= ((grid[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code


def hilbert_encode(grid: torch.Tensor, depth: int) -> torch.Tensor:
    """(M, 3) の非負整数座標をヒルベルト曲線の番号にする（Skilling の方法）。"""
    x = [grid[:, axis].clone() for axis in range(3)]
    top = 1 << (depth - 1)
    q = top
    while q > 1:
        p = q - 1
        for i in range(3):
            has = (x[i] & q) != 0
            if i == 0:
                x[0] = torch.where(has, x[0] ^ p, x[0])
                continue
            t = (x[0] ^ x[i]) & p
            x0 = torch.where(has, x[0] ^ p, x[0] ^ t)
            x[i] = torch.where(has, x[i], x[i] ^ t)
            x[0] = x0
        q >>= 1
    for i in range(1, 3):
        x[i] = x[i] ^ x[i - 1]
    t = torch.zeros_like(x[0])
    q = top
    while q > 1:
        t = torch.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x = [v ^ t for v in x]
    code = torch.zeros_like(x[0])
    for bit in range(depth - 1, -1, -1):
        for i in range(3):
            code = (code << 1) | ((x[i] >> bit) & 1)
    return code


def serialize(grid: torch.Tensor, batch: torch.Tensor, order: str, depth: int) -> torch.Tensor:
    """サンプル番号を上位ビットに置いた直列化の番号。並べ替えるとサンプルごとにまとまる。"""
    if order.endswith("-trans"):
        grid = grid[:, [1, 0, 2]]
    if order.startswith("z"):
        code = z_order_encode(grid, depth)
    elif order.startswith("hilbert"):
        code = hilbert_encode(grid, depth)
    else:
        raise ValueError(f"order は {', '.join(ORDERS)} のどれかです。")
    return (batch << (3 * depth)) | code


def grid_depth(grid: torch.Tensor) -> int:
    return max(1, int(grid.max()).bit_length())
