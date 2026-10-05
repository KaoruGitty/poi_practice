# LiDAR 点群と身体状態から速度指令を回帰する

差動二輪ロボットが、多層 LiDAR の点群と今の身体状態から、次の並進速度 `v` と旋回角速度 `ω` を出す実験です。学習は教師ありの回帰で、評価は同じシミュレータの閉ループです。

## タスク

ロボット座標の点群だけでは、今どれだけ動いているかと、どこへ行きたいかが分かりません。同じ壁が見えていても、止まっているときと 1 m/s で近づいているときでは出すべき指令が違います。加速度には上限があるので、状態を入れない回帰はブレーキのタイミングを誤りやすいです。

| 入力 | 内容 |
| --- | --- |
| 点群 `(288, 3)` | 4 仰角 × 72 方位。ロボット座標のメートル。センサ高さ 0.5 m |
| 状態 `5` | `v`、`ω`、目標の前後位置、左右位置、距離。それぞれ上限で割った値 |
| 出力 `2` | 次の `v` と `ω`。同じく上限で割った値 |

高さ 0.3 m 前後のパレットは、水平ビームの下をすり抜けます。下向きのリングだけが当たります。

ネットワークは、点群を 1 本のベクトルにする**エンコーダ**、状態を小さな全結合でベクトルにする**状態の枝**、2 本を連結して `v` と `ω` を出す**回帰ヘッド**に分かれています。エンコーダは PointNet++ と Point Transformer V3（PTv3）から `--encoder` で選びます。点数はフレームごとに違ってよく、固定点数に揃える必要はありません。構成は [docs/architecture.md](docs/architecture.md)、PointNet++ の中身は [docs/pointnet2.md](docs/pointnet2.md)、PTv3 の中身は [docs/ptv3.md](docs/ptv3.md) にあります。

教師は地図を知っている A* と pure pursuit です。学生が見るのは点群と状態だけです。集めた軌跡の上だけで合わせると、少し外れた瞬間に壁へ向かったまま止まります。そこで、学習済みの方針を混ぜて走らせ、その状態にも教師の指令を付け直して追加します。

## 準備

想定環境は **JetPack 6.x**（Ubuntu 22.04 / **Python 3.10** / **CUDA 12.x**）です。PyTorch は **2.2 以上**（Jetson AI Lab の jp6 向け wheel を推奨）。

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
# Jetson では先に CUDA 付き torch を入れる（JetPack 6.2 / CUDA 12.6 の例）
.venv/bin/pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
.venv/bin/pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
```

CUDA のマイナーが違うときは index を合わせます（例: CUDA 12.8 → `jp6/cu128`、12.9 → `jp6/cu129`）。デスクトップや CPU だけの環境では、torch の行を省いて `pip install -r requirements.txt` だけで構いません。`scripts/prepare.sh` は Jetson を検出すると同じ index から torch を入れます（上書きは `JETSON_TORCH_INDEX`）。

教師データの収集、学習、閉ループの順です。

```bash
.venv/bin/python -m physical_pointnet.sim.geometry
.venv/bin/python -m physical_pointnet.collect --episodes 48 --seed 0 --out data/demos.npz
.venv/bin/python -m physical_pointnet.train --data data/demos.npz --epochs 30 --encoder pointnet2
.venv/bin/python -m physical_pointnet.collect --episodes 36 --seed 1 --ckpt runs/pointnet2/best.pt \
  --append data/demos.npz --out data/demos_dagger.npz
.venv/bin/python -m physical_pointnet.train --data data/demos_dagger.npz --epochs 20 --encoder pointnet2 --out runs/pointnet2_dagger
.venv/bin/python -m physical_pointnet.evaluate --ckpt runs/pointnet2_dagger/best.pt --episodes 8 --gif runs/pointnet2_dagger/rollout.gif
```

エンコーダを PTv3 にするときは `--encoder ptv3` です。保存先を省くと `runs/<エンコーダ名>/` になります。エンコーダと前処理の細かい値は JSON で上書きします。屋外 LiDAR 向けの例を `configs/` に置いています。

```bash
.venv/bin/python -m physical_pointnet.train --data data/outdoor.npz --encoder ptv3 \
  --encoder-config configs/encoders/ptv3_outdoor.json \
  --preprocess-config configs/preprocess/outdoor.json --device cuda --amp
```

状態を外した比較は `--no-status`、増強を切るのは `--no-augment` です。教師だけの走行は `evaluate --expert` です。チェックポイントにはエンコーダの設定と前処理の値も入るので、評価と追加収集は `--ckpt` を渡すだけで同じ構成を組み直します。

`scripts/prepare.sh` は上の一連をまとめて実行します。`ENCODER=ptv3 scripts/prepare.sh` で PTv3 版になり、閉ループ用の重みは `runs/<エンコーダ名>_final/best_closed.pt` です。

## 今回の走行

教師 48 本はすべて目標に着きました。サンプルは 8313 ステップです。その軌跡だけで合わせた方針は、壁の近くで角速度を出し切れず停滞しました。学習済み方針を 75% 混ぜて走らせ、ラベルは教師のまま付け直す収集を 2 回足し、`data/demos_dagger2.npz`（20385 ステップ）にしています。

この節の数字と `runs/` の重みは、エンコーダとヘッドを分ける前の実装で学習したものです。今の実装では旧形式のチェックポイントを読み込めないので、同じ比較をするには学習し直します。新しい PointNet++ と PTv3 は、まだこのデータで学習していません。

PointNet++ はこのデータで 15 エポック学習しました。角速度の二乗誤差は 3 倍しています。閉ループ用の重みは検証中の 4 本で到達率が最も高かった `runs/pointnet2_final/best_closed.pt`（15 エポック目）です。同じ初期値で 8 本ずつ走らせた結果は次です。

| 開始乱数 | 到達 | 衝突 | 停滞 | 時間切れ |
| --- | --- | --- | --- | --- |
| 1000 | 8 本中 4 本 | 1 本 | 1 本 | 2 本 |
| 7 | 8 本中 6 本 | 1 本 | 1 本 | 0 本 |

動画は `runs/pointnet2_final/rollout_room.gif` と `runs/pointnet2_final/rollout_corridor.gif` です。切り替え前の PointNet は同じ手順で 16 本中 13 本が到達し、衝突は 0 本でした。その動画は `runs/pointnet_final/` に残しています。

## 地図

部屋、クランク通路、倉庫の棚、スラロームの 4 種類です。ロボット半径は 0.30 m、最高速度は 1.0 m/s、最高角速度は 1.5 rad/s です。速度と角速度は指令へ一度に追いつかず、加速度で制限されます。

## ファイル

- `physical_pointnet/sim/` 地図、レイキャスト、車両、教師、描画
- `physical_pointnet/preprocess.py` 1 スキャンの前処理（欠損、距離と高さの範囲、ロボット座標への変換、ボクセル間引き、点数の上限、特徴の倍率、学習時の増強）。別の点群データの調べ方は [docs/pointcloud_data.md](docs/pointcloud_data.md)
- `physical_pointnet/points.py` 点数がばらばらのバッチ `PointBatch` と、最遠点サンプリング、近傍探索などの点群演算
- `physical_pointnet/encoders/` 点群エンコーダ。`pointnet2.py`（複数半径のセット抽象化）、`ptv3.py`（直列化と区画の注意）、PTv3 用の `serialization.py` と `sparse_conv.py`
- `physical_pointnet/model.py` 状態の枝、回帰ヘッド、組み立て、チェックポイントの保存と読み込み
- `physical_pointnet/collect.py` 教師デモ、または学習済み方針を混ぜた追加収集
- `physical_pointnet/train.py` 回帰。検証損失の最良と閉ループ成功率が最も高い重みを保存します
- `physical_pointnet/evaluate.py` 閉ループと GIF
- `configs/` 屋外 LiDAR 向けのエンコーダと前処理の設定例
- `scripts/make_doc_figures.py` `docs/images/` の図を、シミュレータのスキャンと実際のエンコーダの関数から作り直す

点群はロボット座標のメートル、状態と指令はスケール済みで `npz` に入ります。列は `points`、`status`、`action`、`episode_id`、`map_id` です。フレームごとに点数が違うデータは、`points` を配列のリスト（`npz` ではオブジェクト配列）にします。

## 参考文献

- Qi, C. R., Su, H., Mo, K., Guibas, L. J. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR 2017.
- Qi, C. R., Yi, L., Su, H., Guibas, L. J. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeurIPS 2017.
- Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., Zhao, H. Point Transformer V3: Simpler, Faster, Stronger. CVPR 2024.
- Ross, S., Gordon, G., Bagnell, J. A. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011.
