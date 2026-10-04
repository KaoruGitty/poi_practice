# 方針ネットワークの構成

点群と身体状態から、次の並進速度 `v` と角速度 `ω` を回帰します。ネットワークは 3 つの部品に分かれています。

1. **点群エンコーダ**。1 スキャンの点群を 1 本のベクトルにします。PointNet++ と Point Transformer V3（PTv3）があり、名前で切り替えます。
2. **状態の枝**。速度や目標位置などの 5 成分を、小さな全結合でベクトルにします。
3. **回帰ヘッド**。2 本のベクトルを横に並べ（連結）、全結合で `v / v_max` と `ω / ω_max` を出します。

状態の枝と回帰ヘッドは、どのエンコーダでも同じです。エンコーダを替えても、学習、評価、データ収集のコードは変わりません。

```text
LiDAR 1 スキャン (N, 3 + E)      ← N はフレームごとに違ってよい
        │  preprocess_scan（欠損、範囲、座標変換、ボクセル、点数上限、倍率）
        ▼
coord (n, 3) メートル ／ feat (n, 3 + E)
        │  サンプルを縦につないで PointBatch に
        ▼
点群エンコーダ  pointnet2 か ptv3 ──────► (B, D)
                                           │
状態 (B, 5) ──► 状態の枝 5→64→64 ──► (B, 64)
                                           │
                                  連結 (B, D + 64)
                                           ▼
                       回帰ヘッド  →128 → dropout 0.2 → 2
                                           ▼
                                  v / v_max ,  ω / ω_max
```

`D` は既定の設定で、PointNet++ が 512、PTv3 が 256 です。

## ファイル

| ファイル | 中身 |
| --- | --- |
| `physical_pointnet/preprocess.py` | 1 スキャンをエンコーダの入力にそろえる `preprocess_scan` と、その設定 `PreprocessConfig`、学習時の増強 `AugmentConfig` |
| `physical_pointnet/points.py` | 点数がばらばらのバッチ `PointBatch` と、最遠点サンプリング、半径内の近傍探索、サンプルごとの max と平均 |
| `physical_pointnet/encoders/base.py` | エンコーダの約束事 `PointEncoder` |
| `physical_pointnet/encoders/pointnet2.py` | PointNet++（複数半径のセット抽象化） |
| `physical_pointnet/encoders/ptv3.py` | PTv3 本体 |
| `physical_pointnet/encoders/serialization.py` | PTv3 用の Z 曲線とヒルベルト曲線 |
| `physical_pointnet/encoders/sparse_conv.py` | PTv3 用の疎な 3D 畳み込み（PyTorch だけで書いたもの） |
| `physical_pointnet/encoders/__init__.py` | 名前からエンコーダを作る `build_encoder` |
| `physical_pointnet/model.py` | 状態の枝、回帰ヘッド、組み立て、チェックポイント |

## 点数がばらばらのバッチ

屋外の LiDAR は、フレームごとに点数が違います。固定点数に揃えると、多いフレームは捨て、少ないフレームは複製することになります。このリポジトリでは揃えずに、サンプルを縦につないで渡します。PyG や Pointcept と同じ考え方です。

```16:27:physical_pointnet/points.py
@dataclass
class PointBatch:
    """全サンプルの点を縦につないだバッチ。

    batch[i] は点 i が属するサンプル番号で、昇順に並んでいる必要がある。
    各サンプルには少なくとも 1 点ある。
    """

    coord: torch.Tensor  # (M, 3) ロボット座標のメートル
    feat: torch.Tensor  # (M, C) 点ごとの入力特徴
    batch: torch.Tensor  # (M,) long
    batch_size: int
```

3 つのサンプルが 2 点、3 点、1 点なら、`batch` は `[0, 0, 1, 1, 1, 2]` です。`PointBatch.from_arrays(coords, feats)` に numpy 配列のリストを渡すと、この形になります。

`coord` は距離の計算に使う生のメートルです。`feat` は倍率を掛けた xyz と、強度などの追加の列です。半径や格子の大きさはメートルで指定できるので、座標をどう割ったかを気にせずに決められます。

PointNet++ は内部でサンプルごとの表（足りない分はマスク）に直してから計算します。PTv3 は縦につないだまま計算します。

## エンコーダの約束事

```11:17:physical_pointnet/encoders/base.py
class PointEncoder(nn.Module):
    """PointBatch を受け取り、サンプルごとに (B, out_channels) のベクトルを返す。"""

    out_channels: int

    def forward(self, points: PointBatch) -> torch.Tensor:  # pragma: no cover - 抽象
        raise NotImplementedError
```

エンコーダは `PointBatch` を受け取り、`(B, out_channels)` を返します。`out_channels` は作った時点で決まっていて、回帰ヘッドの入力の幅はそこから計算します。新しいエンコーダを足すときは、この形で書いて `encoders/__init__.py` の `ENCODERS` に名前を登録します。

```9:18:physical_pointnet/encoders/__init__.py
ENCODERS: dict[str, type[PointEncoder]] = {
    "pointnet2": PointNet2Encoder,
    "ptv3": PTv3Encoder,
}


def build_encoder(name: str, in_channels: int, **kwargs) -> PointEncoder:
    if name not in ENCODERS:
        raise ValueError(f"エンコーダは {', '.join(ENCODERS)} のどれかです。今は {name!r} です。")
    return ENCODERS[name](in_channels=in_channels, **kwargs)
```

## エンコーダの切り替え

学習のコマンドで `--encoder` を替えるだけです。

```bash
.venv/bin/python -m physical_pointnet.train --data data/demos.npz --encoder pointnet2
.venv/bin/python -m physical_pointnet.train --data data/demos.npz --encoder ptv3
```

エンコーダの細かい値は JSON で渡します。書いた項目だけが既定値を上書きします。屋外向けの例を `configs/encoders/` に置いています。

```bash
.venv/bin/python -m physical_pointnet.train --data data/outdoor.npz --encoder ptv3 \
  --encoder-config configs/encoders/ptv3_outdoor.json \
  --preprocess-config configs/preprocess/outdoor.json --device cuda --amp
```

コードから作るときは次のとおりです。

```python
from physical_pointnet.model import ModelConfig, build_model

model = build_model(ModelConfig(encoder="ptv3", encoder_kwargs={"grid_size": 0.05}, in_channels=4))
```

`in_channels` は `feat` の幅で、`PreprocessConfig.in_channels`（3 + 追加の列の数）と一致させます。学習スクリプトは自動で合わせます。

## 状態の枝と回帰ヘッド

```59:94:physical_pointnet/model.py
class StatusEncoder(nn.Module):
    """身体状態（速度、目標位置など）をベクトルにする。"""

    def __init__(self, status_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.net = _mlp([status_dim, *hidden])
        self.out_channels = hidden[-1] if hidden else status_dim

    def forward(self, status: torch.Tensor) -> torch.Tensor:
        return self.net(status)


class RegressionHead(nn.Module):
    """連結したベクトルから連続値を出す。最後の層に活性化は付けない。"""

    def __init__(self, in_channels: int, hidden: tuple[int, ...], out_dim: int, dropout: float):
        super().__init__()
        self.body = _mlp([in_channels, *hidden], dropout)
        self.out = nn.Linear(hidden[-1] if hidden else in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class PolicyModel(nn.Module):
    def __init__(self, encoder: nn.Module, status_encoder: StatusEncoder | None, head: RegressionHead):
        super().__init__()
        self.encoder = encoder
        self.status_encoder = status_encoder
        self.head = head

    def forward(self, points: PointBatch, status: torch.Tensor) -> torch.Tensor:
        z = self.encoder(points)
        if self.status_encoder is not None:
            z = torch.cat([z, self.status_encoder(status).to(z.dtype)], dim=1)
        return self.head(z)
```

状態は点の特徴には混ぜず、エンコーダが点群を 1 本にしたあとで連結します（後段での結合、late fusion）。近傍の取り方や注意の重みは点の幾何だけで決まり、速度と目標は大域特徴の横に一度だけ置かれます。`--no-status` では状態の枝がなくなり、エンコーダの出力だけがヘッドに入ります。

状態の 5 成分は次のとおりで、すべて上限で割った無次元の値です。目標はロボット座標（x が前、y が左）です。

| 成分 | 中身 | 割る値 |
| --- | --- | --- |
| 0 | 今の並進速度 `v` | 1.0 m/s |
| 1 | 今の角速度 `ω` | 1.5 rad/s |
| 2 | 目標の前後位置 | 12 m |
| 3 | 目標の左右位置 | 12 m |
| 4 | 目標までの距離 | 12 m |

ヘッドの出力に活性化はありません。学習ラベルは `v / v_max`（0〜1）と `ω / ω_max`（-1〜1）です。閉ループでは、範囲をはみ出した値だけを端に戻してから物理量にします。損失は並進の二乗誤差に、旋回の二乗誤差を `--omega-weight` 倍して足します。

幅や dropout は `ModelConfig` の `status_hidden`、`head_hidden`、`dropout` で変えます。

## チェックポイント

`best.pt` などには、重みと一緒に 3 つの設定を保存します。

| キー | 中身 |
| --- | --- |
| `version` | 2 |
| `model` | `ModelConfig`。エンコーダの名前と設定、状態の枝とヘッドの幅 |
| `preprocess` | `PreprocessConfig`。学習で使った前処理の値 |
| `sim` | `SimConfig`。速度の上限など |
| `state_dict` | 重み |

`load_checkpoint` はこの設定からネットワークを組み直し、推論で同じ前処理を使います。エンコーダを分ける前の形式（`runs/pointnet2_final/` などに残っている重み）は読み込めず、学習し直すよう伝えるエラーになります。

## PointNet++ エンコーダ

ステップごとの詳しい説明は [pointnet2.md](pointnet2.md) にあります。

[PointNet++](https://arxiv.org/abs/1706.02413) のセット抽象化を、複数半径（MSG、multi-scale grouping）で重ねています。分類用の公開モデル `pointnet2_cls_msg`（[charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)、PyTorch 移植の [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)）と同じ組み立てで、最後のクラス分類の全結合だけを持ちません。

```91:99:physical_pointnet/encoders/pointnet2.py
    def forward(self, points: PointBatch) -> torch.Tensor:
        xyz, mask = to_padded(points.coord, points.batch, points.batch_size)
        feat, _ = to_padded(points.feat, points.batch, points.batch_size)
        for layer in self.layers:
            xyz, feat, mask = layer(xyz, feat, mask, random_start=self.training)
        grouped = torch.cat([xyz * self.coord_scale, feat], dim=-1)
        hidden = grouped.new_full((*mask.shape, self.out_channels), float("-inf"))
        hidden[mask] = self.global_mlp(grouped[mask]).to(hidden.dtype)
        return hidden.max(dim=1).values
```

各段でやることは公開モデルと同じ順番です。

1. 残す点を最遠点サンプリングで選ぶ。学習中は始点をランダムにし、推論では先頭の点から始める。
2. 選んだ点のまわりで、半径ごとに近い点を最大 `nsample` 個取る。
3. 中心からの相対位置を半径で割り、前段の特徴と連結する。
4. 点ごとに同じ全結合（Linear、バイアスなし、BatchNorm、ReLU）を掛ける。
5. グループ内でチャネルごとに max を取る。半径ごとの結果を横に並べ、次の段の特徴にする。

最後に、残った点の位置（`coord_scale` を掛けたもの）と特徴に全結合を掛け、全点の max を取って 1 本にします。

公開実装との違いは次のとおりです。

| | 公開実装（yanx27） | このリポジトリ |
| --- | --- | --- |
| 点数 | バッチ内で同じ。足りなければ複製 | サンプルごとに違ってよい。足りない分はマスク |
| 半径の単位 | 単位球に縮めた座標 | メートル |
| 相対位置 | そのまま | 半径で割る（どの段でも 1 前後） |
| 半径内の点が `nsample` に満たないとき | 最初に見つかった点を繰り返す | 余りは max の対象から外す |
| 全結合 | `Conv2d`（1×1） | `Linear` と `BatchNorm1d`（同じ計算） |
| 出力 | クラスの対数確率 | 大域特徴 `(B, D)` |

既定の値（`DEFAULT_SA_LAYERS`）は、このリポジトリのシミュレータ（1 スキャン 300 点弱、最大 8 m）に合わせています。

| 段 | 残す点 | 半径 (m) | 近傍の数 | 全結合 |
| --- | --- | --- | --- | --- |
| 1 | 64 | 0.8、1.6 | 8、16 | 3+3→32→32→64、3+3→64→64→128 |
| 2 | 16 | 2.0、4.0 | 8、16 | 192+3→64→64→128、192+3→128→128→256 |
| 全体 | 1 | — | — | 384+3→256→512 |

屋外向けの例 [`configs/encoders/pointnet2_outdoor.json`](../configs/encoders/pointnet2_outdoor.json) は 3 段（4096、1024、256 点）で、半径は 0.5〜4 m、出力は 1024 次元です。

## PTv3 エンコーダ

詳しくは [ptv3.md](ptv3.md) にあります。要点は次のとおりです。

- 点をボクセルにまとめ、空間充填曲線の順に並べます。並べた順に `patch_size` 点ずつ区切り、その中で自己注意を掛けます。近傍探索がいらないので、点が多くても速く動きます。
- 位置の情報は、注意の前に置く疎な 3D 畳み込みで入れます。
- 段の間でボクセルを粗くし、最後に全点の max（または max と平均）で 1 本にします。

## 別のデータで値を決めるときに触る場所

| 何を合わせるか | 場所 |
| --- | --- |
| センサの取り付け位置と向き | `PreprocessConfig.sensor_rotation`、`sensor_translation` |
| 使う距離と高さの範囲 | `max_range`、`min_range`、`z_min`、`z_max` |
| 点の密度と上限 | `voxel_size`、`max_points` |
| 入力チャネル（強度など） | `extra_channels`、`extra_scales`。`in_channels` は自動で決まる |
| 特徴の座標の倍率 | `coord_scale_xy`、`coord_scale_z`。値が 1 前後に収まるように |
| PointNet++ の近所の広さ | `sa_layers` の `radii`（メートル）と `nsamples` |
| PointNet++ の点の減らし方 | `sa_layers` の `npoint`。各段で 1/4 前後が目安 |
| PTv3 の最小の解像度 | `grid_size`（メートル）。前処理の `voxel_size` と同じか少し大きく |
| PTv3 の注意の範囲 | `enc_patch_size`。大きいほど遠くまで見るが、重くなる |
| 状態の中身 | `obs.make_status` と `ModelConfig.status_dim` |
| 出力の数 | `ModelConfig.out_dim` と、`obs.normalize_action` / `denormalize_action` |

左右反転の増強を使うときは、状態とラベルのどの成分の符号を反転するかを `obs.STATUS_MIRROR_SIGN` と `obs.ACTION_MIRROR_SIGN` に書きます。
