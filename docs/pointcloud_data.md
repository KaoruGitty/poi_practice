# もらった点群データの調べ方と前処理

点群データを受け取ったら、学習の前に次の5点を確定させます。

1. ファイル形式と、1点あたりの列（xyz のほかに強度、リング番号、時刻、ラベルなど）
2. 単位（メートルかミリか）と、座標の原点・軸の向き
3. 1フレームの点数と、それがフレームごとに変わるか
4. 欠損値（NaN、0 埋め、無限大）の入り方
5. ラベルや状態（速度、関節角など）が、どのフレームとどう対応しているか

ここで決めたことを前処理に書き、学習と推論で同じ関数を通します。

## 1. 形式を当てる

まず拡張子と中身の先頭を見ます。

```bash
ls -lh data/
find data -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn   # 拡張子ごとの数
file data/sample.pcd
head -c 400 data/sample.pcd | strings | head -20                 # ヘッダがテキストか
xxd data/sample.bin | head                                        # 中身がバイナリか
```

よくある形式と読み方は次のとおりです。

| 拡張子 | よく見る場所 | 中身 | 読み方 |
| --- | --- | --- | --- |
| `.pcd` | PCL、ROS から書き出し | テキストのヘッダに列名と型。本体は ascii か binary | 下の `read_pcd`、または Open3D |
| `.ply` | スキャナ、メッシュ | テキストのヘッダに `element vertex N` と `property` | `plyfile`、Open3D |
| `.bin` | KITTI、nuScenes | ヘッダなし。float32 を並べただけ | `np.fromfile` で列数を当てる |
| `.las` / `.laz` | 航空・地上の測量 | 点フォーマット番号つき。座標はスケールとオフセットつきの整数 | `laspy`（`.laz` は `lazrs` も入れる） |
| `.npy` / `.npz` | 研究コード | numpy 配列そのもの | `np.load` |
| `.h5` / `.hdf5` | ModelNet40 の HDF5 版など | 名前つきの配列 | `h5py` |
| `.txt` / `.csv` / `.xyz` | 簡易な書き出し | 1行1点の数値 | `np.loadtxt` |
| `.bag` / `.db3` / `.mcap` | ROS1、ROS2 | 時刻つきメッセージ。点群は `sensor_msgs/PointCloud2` | `rosbags` |

ヘッダなしの `.bin` は、列数を自分で当てます。KITTI の Velodyne は x, y, z, 反射強度の 4 列です。nuScenes の LiDAR は x, y, z, 強度, リング番号の 5 列です。

```python
import numpy as np

raw = np.fromfile("data/000000.bin", dtype=np.float32)
for cols in (3, 4, 5, 6):
    if raw.size % cols == 0:
        pts = raw.reshape(-1, cols)
        print(cols, pts.shape, pts[:, :3].min(0).round(2), pts[:, :3].max(0).round(2))
```

割り切れる列数のうち、xyz の範囲が数十メートルに収まり、強度の列が 0〜1 か 0〜255 に収まるものが正解です。

### PCD を numpy で読む

Open3D の `read_point_cloud` は xyz と色と法線しか返さず、強度やリング番号は落ちます。全部の列が欲しいときはヘッダを読みます。`binary_compressed` は扱っていないので、その場合は `pypcd4` などを使います。

```python
import numpy as np

_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
              ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_pcd(path: str) -> np.ndarray:
    """列名つきの構造化配列で返す。ascii と binary に対応。"""
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        body = f.read()
    names = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
    dtype = np.dtype([
        (name, _PCD_TYPES[(t, s)], (c,)) if c > 1 else (name, _PCD_TYPES[(t, s)])
        for name, s, t, c in zip(names, sizes, types, counts)
    ])
    n = int(header["POINTS"][0])
    mode = header["DATA"][0]
    if mode == "binary":
        return np.frombuffer(body, dtype=dtype, count=n)
    if mode == "ascii":
        table = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2)
        out = np.empty(len(table), dtype=dtype)
        col = 0
        for name, c in zip(names, counts):
            out[name] = table[:, col] if c == 1 else table[:, col:col + c]
            col += c
        return out
    raise ValueError(f"DATA {mode} は未対応です")


cloud = read_pcd("data/sample.pcd")
print(cloud.dtype.names)          # 例: ('x', 'y', 'z', 'intensity', 'ring')
xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)
```

### LAS / LAZ

```python
import laspy
import numpy as np

las = laspy.read("data/tile.laz")
print(las.header.point_format.id, las.header.point_count)
print(list(las.point_format.dimension_names))   # X, Y, Z, intensity, classification など
xyz = np.stack([las.x, las.y, las.z], axis=1)   # 小文字はスケール適用後の実座標
```

測量データは座標が数十万メートル規模のことが多いです。float32 にする前に、タイルの中心などを引いて桁を落とします。

### ROS bag の PointCloud2

ROS をインストールしなくても `rosbags` で ROS1 と ROS2 の両方を読めます。PointCloud2 は「1点が `point_step` バイトで、各列がそのどの位置にあるか」をメッセージ自身が持っています。

```python
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

_ROS_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2_to_numpy(msg) -> np.ndarray:
    dtype = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [_ROS_TYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")
    return np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)


with AnyReader([Path("data/run.bag")]) as reader:
    for c in reader.connections:
        print(c.topic, c.msgtype, c.msgcount)       # どのトピックに何が入っているか
    lidar = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
    for conn, stamp_ns, raw in reader.messages(connections=lidar):
        msg = reader.deserialize(raw, conn.msgtype)
        cloud = pointcloud2_to_numpy(msg)
        print(stamp_ns, msg.header.frame_id, cloud.dtype.names, len(cloud))
        break
```

`header.frame_id` が点群の座標系の名前です。ロボット本体（`base_link`）との位置関係は `/tf_static` トピックに入っています。

## 2. ファイルを直接開くとこう見える

ここに載せた出力は、すべて実際にコマンドを実行した結果です。リポジトリの `npz` は本物です。ほかの形式は、各形式の仕様どおりに同じ 4 点（前の壁、左の壁、床、返ってこなかったビーム）を書いた小さなファイルです。

| 点 | x | y | z | 強度 | リング |
| --- | --- | --- | --- | --- | --- |
| 前の壁 | 4.125 | 0.031 | 0.512 | 0.62 | 5 |
| 左の壁 | 0.880 | 2.403 | 0.498 | 0.35 | 7 |
| 床 | 1.951 | -0.402 | -0.497 | 0.08 | 0 |
| 欠損 | NaN | NaN | NaN | 0 | 15 |

### `file` コマンド

形式名が分かるのは、先頭に決まった文字列がある形式だけです。ヘッダなしの `.bin` は `data` としか出ません。

```text
$ file scan_ascii.pcd scan_binary.pcd scan.ply 000000.bin scan.csv
scan_ascii.pcd:  Point Cloud Data
scan_binary.pcd: Point Cloud Data
scan.ply:        PLY model, binary, little endian, version 1.0
000000.bin:      data
scan.csv:        CSV text
```

### PCD（ascii）

ヘッダも本体もテキストなので、`cat` でそのまま読めます。

```text
$ cat scan_ascii.pcd
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 4 2
TYPE F F F F U
COUNT 1 1 1 1 1
WIDTH 4
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 4
DATA ascii
4.125 0.031 0.512 0.62 5
0.88 2.403 0.498 0.35 7
1.951 -0.402 -0.497 0.08 0
nan nan nan 0 15
```

ヘッダの読み方です。

- `FIELDS`：列の名前。ここで xyz 以外に何があるかが分かります。
- `SIZE` と `TYPE`：1列のバイト数と型。`F` は浮動小数、`U` は符号なし整数、`I` は符号つき整数です。`ring` は 2 バイトの符号なし整数です。
- `WIDTH` と `HEIGHT`：`HEIGHT 1` なら点を1列に並べたもの。`HEIGHT` が 16 や 64 なら、リング×方位の格子のまま入っています（organized point cloud）。格子のままだと欠損も NaN として残ります。
- `VIEWPOINT`：センサの位置と向き（xyz と四元数）。多くは原点のままです。
- `DATA`：`ascii`、`binary`、`binary_compressed` のどれか。

### PCD（binary）

ヘッダは同じテキストで、`DATA binary` の改行の直後から本体が始まります。本体は人間には読めないので、`xxd` で 16 進数を見ます。

```text
$ xxd scan_binary.pcd | tail -6
000000b0: 2034 0a44 4154 4120 6269 6e61 7279 0a00   4.DATA binary..
000000c0: 0084 40b6 f3fd 3c6f 1203 3f52 b81e 3f05  ..@...<o..?R..?.
000000d0: 00ae 4761 3fc1 ca19 40db f9fe 3e33 33b3  ..Ga?...@...>33.
000000e0: 3e07 005e baf9 3ff2 d2cd bec9 76fe be0a  >..^..?.....v...
000000f0: d7a3 3d00 0000 00c0 7f00 00c0 7f00 00c0  ..=.............
00000100: 7f00 0000 000f 00                        .......
```

1点は 4+4+4+4+2 = 18 バイトです。`0a`（改行）の直後の `00 00 84 40` が最初の x で、リトルエンディアンの float32 として読むと 4.125 です。各点の最後の 2 バイト `05 00`、`07 00`、`00 00`、`0f 00` がリング番号 5、7、0、15 です。`00 00 c0 7f` が3回続くところが、欠損点の NaN です。

本体のバイト数は「点数 × 1点のバイト数」に一致します。一致しなければ、ヘッダの読み違いか、`binary_compressed` です。

### PLY（binary）

ヘッダはテキストで、`end_header` の後ろがバイナリです。`strings` で見ると、ヘッダの後ろに読めない文字が続きます。

```text
$ head -c 220 scan.ply | strings
format binary_little_endian 1.0
comment written by test
element vertex 4
property float x
property float y
property float z
property float intensity
end_header
@���<o
?�Ga?��
@���>33�>^��?��;�v��
```

先頭行の `ply` は短すぎて `strings` には出ませんが、ファイルは必ず `ply` で始まります。`element vertex 4` が点数、`property` の並びが列の順番です。メッシュなら、このあとに `element face` も続きます。

### KITTI 形式の `.bin`

ヘッダがなく、float32 が並んでいるだけです。

```text
$ ls -l 000000.bin
48 バイト

$ xxd 000000.bin
00000000: 0000 8440 b6f3 fd3c 6f12 033f 52b8 1e3f  ...@...<o..?R..?
00000010: ae47 613f c1ca 1940 dbf9 fe3e 3333 b33e  .Ga?...@...>33.>
00000020: 5eba f93f f2d2 cdbe c976 febe 0ad7 a33d  ^..?.....v.....=
```

48 バイト ÷ 4 バイト = 12 個の float32 です。12 は 3 でも 4 でも割り切れるので、サイズだけでは列数が決まりません。両方で並べてみると違いが分かります。

```text
>>> raw.reshape(-1, 4)          # 正しい
[[ 4.125  0.031  0.512  0.62 ]
 [ 0.88   2.403  0.498  0.35 ]
 [ 1.951 -0.402 -0.497  0.08 ]]

>>> raw.reshape(-1, 3)[:2]      # 列数を間違えた
[[4.125 0.031 0.512]
 [0.62  0.88  2.403]]
```

列数を間違えると、2行目の先頭に前の点の強度（0.62）が x として入り、そこから全部ずれます。z の列が地面の高さ付近にまとまらず、点群を描くと形が崩れるので気づけます。

### CSV / テキスト

```text
$ cat scan.csv
x,y,z,intensity
4.125,0.031,0.512,0.620
0.880,2.403,0.498,0.350
1.951,-0.402,-0.497,0.080
```

1行目が列名のことも、いきなり数字のこともあります。`np.loadtxt(path, delimiter=",", skiprows=1)` で読みます。区切りが空白なら `delimiter` を外します。

### このリポジトリの `npz`（実データ）

`npz` は zip の中に `.npy` が並んだものです。中身の一覧は `unzip -l` でも見えます。

```text
$ unzip -l data/demos_dagger2.npz
  Length      Date    Time    Name
---------  ---------- -----   ----
 70450688  01-01-1980 00:00   points.npy
   407828  01-01-1980 00:00   status.npy
   163208  01-01-1980 00:00   action.npy
    81668  01-01-1980 00:00   episode_id.npy
    81668  01-01-1980 00:00   map_id.npy
      132  01-01-1980 00:00   v_max.npy
      132  01-01-1980 00:00   w_max.npy
      132  01-01-1980 00:00   lidar_max_range.npy
      132  01-01-1980 00:00   z_scale.npy
```

numpy で開くと、形と型が分かります。

```text
>>> d = np.load("data/demos_dagger2.npz")
points           shape=(20385, 288, 3)  dtype=float32
status           shape=(20385, 5)       dtype=float32
action           shape=(20385, 2)       dtype=float32
episode_id       shape=(20385,)         dtype=int32
map_id           shape=(20385,)         dtype=int32
v_max            shape=()               dtype=float32
w_max            shape=()               dtype=float32
lidar_max_range  shape=()               dtype=float32
z_scale          shape=()               dtype=float32
```

最初のフレームの点を見ます。

```text
>>> d["points"][0, :6]          # 一番下向きのリング
[[-2.352 -0.     0.   ]
 [-2.343 -0.205  0.   ]
 [-2.317 -0.408  0.   ]
 [-2.272 -0.609  0.   ]
 [-2.21  -0.805  0.   ]
 [-2.132 -0.994  0.   ]]

>>> d["points"][0, 72:75]       # 2 本目のリング
[[-6.06  -0.     0.288]
 [-7.074 -0.619  0.252]
 [-7.874 -1.388  0.221]]

>>> d["status"][0]
[ 0.     0.    -0.797 -0.289  0.847]

>>> d["action"][0]
[ 0.25 -1.  ]
```

この数字から読めることです。

- 最初の 72 点は z がすべて 0 です。下向き 12 度のリングが床に当たっています。z の原点が床です。
- 1 点目は x が負で y が 0 なので、方位の並びは真後ろから始まっています。
- 73 点目からが次のリングです。72 点ごとにリングが変わります。
- 水平距離は最小 1.79 m、最大 8.00 m でした。8 m は最大距離で、何にも当たらなかったビームがこの値で止まっています。欠損を NaN ではなく最大距離で表す形式です。
- `status` は、止まっていて（速度 0、角速度 0）、目標が後ろ（前後位置が負）の右側（左右位置が負）にある状態です。
- `action` は、ゆっくり前へ出ながら、右へ最大の角速度で回る指令です。

### ROS bag と LAS

この2つは、手元に小さな例を作れる道具が入っていないので出力は載せていません。どちらも中身を直接見るより、ライブラリで一覧を出すのが早いです。

- ROS bag：`rosbags` で `reader.connections` を出すと、トピック名、型、メッセージ数が並びます（1 章のコード）。PointCloud2 の1メッセージには `fields`（列名、バイト位置、型）、`point_step`（1点のバイト数）、`width` と `height`、`is_dense`（欠損なしか）、`data`（本体のバイト列）が入っています。
- LAS：ファイルは必ず `LASF` の4文字で始まるので、`head -c 4` で確かめられます。中身は `laspy` で `header.point_format.id` と `point_format.dimension_names` を出すと分かります。

## 3. 中身を数字で確かめる

読めたら、1フレームずつではなく、ランダムに選んだ数十フレームで統計を取ります。

```python
import numpy as np


def describe(frames: list[np.ndarray], names: list[str]) -> None:
    """frames は (N_i, C) の配列のリスト。names は列名。"""
    counts = np.array([len(f) for f in frames])
    print(f"フレーム数={len(frames)}  点数 min={counts.min()} 中央={int(np.median(counts))} max={counts.max()}")
    allp = np.concatenate(frames, axis=0)
    bad = ~np.isfinite(allp).all(axis=1)
    zero = (allp[:, :3] == 0).all(axis=1)
    print(f"NaN/inf を含む点={bad.mean():.1%}  原点ちょうどの点={zero.mean():.1%}")
    good = allp[~bad]
    for i, name in enumerate(names):
        col = good[:, i]
        q = np.percentile(col, [0, 1, 50, 99, 100])
        print(f"{name:>10s}  " + "  ".join(f"{v:9.3f}" for v in q))
    r = np.linalg.norm(good[:, :2], axis=1)
    print(f"水平距離 中央={np.median(r):.2f}  99%={np.percentile(r, 99):.2f}")
```

読み取り方の目安です。

- **単位**：屋内 LiDAR で水平距離の 99% が数千なら、ミリメートルです。1000 で割ります。物体データで最大が 1 前後なら、すでに正規化済みです。
- **欠損**：NaN や原点ちょうどの点は「返ってこなかったビーム」のことが多いです。`is_dense=False` の PointCloud2 もこれを含みます。距離が最大値ちょうどの点がまとまってあるなら、欠損を最大距離で埋める形式です（このリポジトリの `npz` がそうです）。
- **点数**：毎回同じなら、リング×方位の格子のまま入っています。変わるなら、欠損を落としたあとのデータです。
- **軸の向き**：z の 1% 点が地面の高さに近いはずです。センサ原点の点群なら、地面はおよそ「マイナスの取り付け高さ」に出ます。前方に壁がある場面で x が正なら、x が前です。
- **強度**：0〜1、0〜255、0〜65535 のどれかで、センサによって違います。

最後に必ず絵で見ます。上から見た図と横から見た図で、軸の向きと地面の高さが分かります。

```python
import matplotlib

matplotlib.use("Agg")   # 画面なしでファイルに保存する
import matplotlib.pyplot as plt


def plot_frame(xyz, path="frame.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(xyz[:, 0], xyz[:, 1], s=1, c=color if color is not None else xyz[:, 2], cmap="viridis")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect("equal"); axes[0].set_title("top")
    axes[1].scatter(xyz[:, 0], xyz[:, 2], s=1, c="k")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z"); axes[1].set_title("side")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
```

## 4. ラベルや状態との対応を確かめる

点群と一緒に教師（速度指令、関節角、クラス）があるときは、どの時刻の値をそのフレームに付けるかを決めます。センサごとに周期が違うのが普通です（LiDAR 10 Hz、オドメトリ 30 Hz、ジョイスティック 不定期 など）。

```python
import numpy as np


def nearest_before(t_query: np.ndarray, t_src: np.ndarray, values: np.ndarray, max_gap: float):
    """各 t_query より前で一番近い t_src の値。max_gap 秒より古ければ無効。"""
    order = np.argsort(t_src)
    t_src, values = t_src[order], values[order]
    idx = np.searchsorted(t_src, t_query, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(t_src) - 1)
    valid &= (t_query - t_src[idx]) <= max_gap
    return values[idx], valid
```

現在の状態（今の速度など）は「点群より前で一番近い値」にします。後の値を使うと、推論時には手に入らない未来の情報が混ざります。行動のラベル（次に出す指令）は、点群の直後に出た指令を使うことが多いです。

## 5. 前処理

ここからは、確かめた内容に合わせて毎フレーム同じ処理を通します。順番は次がおすすめです。

このリポジトリでは、この節の処理を [`physical_pointnet/preprocess.py`](../physical_pointnet/preprocess.py) の `preprocess_scan` にまとめています。値は `PreprocessConfig` で指定します。学習（`dataset.py`）と閉ループ（`evaluate.py`）の両方が同じ関数を通ります。設定はチェックポイントに保存されるので、推論側で値を合わせ直す必要はありません。下の短いコードは、それぞれの処理が何をしているかを示すための単体の例です。

| 処理 | `PreprocessConfig` の項目 |
| --- | --- |
| 欠損を落とす | 常に行う（NaN と inf）。`drop_no_return` で最大距離ちょうどの点も落とす |
| 範囲を切る（距離） | `max_range`、`min_range`、`range_origin`（入力座標でのセンサ位置） |
| ロボット座標へ | `sensor_rotation`（行優先の 3×3）、`sensor_translation` |
| 範囲を切る（高さ） | `z_min`、`z_max`（ロボット座標） |
| ボクセル間引き | `voxel_size`（0 なら行わない） |
| 点数の上限 | `max_points`、`sampling`（`random` か `fps`） |
| 特徴の倍率 | `coord_scale_xy`、`coord_scale_z`、`extra_channels`、`extra_scales` |
| 学習時の増強 | `AugmentConfig`（揺らし、間引き、左右反転） |

屋外 LiDAR の設定例は [`configs/preprocess/outdoor.json`](../configs/preprocess/outdoor.json) です。`--preprocess-config` で渡すと、シミュレータ用の既定値の上に書いた項目だけが上書きされます。

### 欠損を落とす

```python
def drop_invalid(points: np.ndarray) -> np.ndarray:
    ok = np.isfinite(points).all(axis=1) & ~(points[:, :3] == 0).all(axis=1)
    return points[ok]
```

### 単位と座標系をそろえる

センサ座標からロボット座標へは、取り付け位置の回転 `R` と並進 `t` を掛けます。値は ROS なら `/tf_static`、データセットなら calibration ファイルにあります。

```python
def to_robot_frame(xyz: np.ndarray, R: np.ndarray, t: np.ndarray, unit_scale: float = 1.0) -> np.ndarray:
    return (xyz * unit_scale) @ R.T + t
```

このリポジトリでは「x が前、y が左、z が上、原点はロボットの真下の床」です。別データもこの向きにそろえると、状態の目標位置などと同じ座標で扱えます。

### 範囲を切る

学習に使う範囲だけを残します。遠すぎる点は数が少なく、ほぼノイズです。

```python
def crop(points: np.ndarray, max_range: float, z_min: float, z_max: float) -> np.ndarray:
    r = np.linalg.norm(points[:, :2], axis=1)
    keep = (r <= max_range) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[keep]
```

床の点を落とすかどうかはタスク次第です。このリポジトリでは低い障害物の手前に床が見えること自体が手がかりなので、床も残しています。

### 密度をならす（ボクセル間引き）

近くは点が密で、遠くは疎です。一定の大きさの格子ごとに1点だけ残すと、近くに偏った重みが減ります。

```python
def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]
```

### 点数をそろえる

公開されている PointNet++ の多くの実装は、バッチの点数を揃える必要があります。多ければ最遠点サンプリングかランダムで減らし、少なければ既存の点を複製して増やします。最大値プーリングなので、複製した点は結果を変えません。

このリポジトリのエンコーダは点数が揃っていなくても動きます。サンプルを縦に並べ、各点がどのサンプルのものかを番号で持つ形（`points.PointBatch`）だからです。そのため `preprocess_scan` は点数を揃えず、上限 `max_points` を超えたときだけ減らします。下の `fix_count` は、固定点数の公開実装に入れるときの例です。

```python
def farthest_point_sample(xyz: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.zeros(n, dtype=np.int64)
    dist = np.full(len(xyz), np.inf)
    idx[0] = rng.integers(len(xyz))
    for i in range(1, n):
        dist = np.minimum(dist, ((xyz - xyz[idx[i - 1]]) ** 2).sum(axis=1))
        idx[i] = int(dist.argmax())
    return idx


def fix_count(points: np.ndarray, n: int, rng: np.random.Generator, fps: bool = True) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("有効な点がありません")
    if len(points) >= n:
        idx = farthest_point_sample(points[:, :3], n, rng) if fps else rng.choice(len(points), n, replace=False)
    else:
        extra = rng.choice(len(points), n - len(points), replace=True)
        idx = np.concatenate([np.arange(len(points)), extra])
    return points[rng.permutation(idx)]
```

最遠点サンプリングは数万点だと遅いので、先にボクセル間引きかランダムで数千点まで落としてから使います。

### 正規化

物体単位とシーン単位で方針が違います。

```python
def normalize_object(xyz: np.ndarray) -> np.ndarray:
    """物体を中心へ寄せ、半径 1 の球に入れる。ModelNet と同じ。"""
    centered = xyz - xyz.mean(axis=0)
    return centered / np.linalg.norm(centered, axis=1).max()


def normalize_scene(xyz: np.ndarray, max_range: float, z_scale: float) -> np.ndarray:
    """原点はロボットのまま。距離の上限で割る。"""
    return xyz / np.array([max_range, max_range, z_scale], dtype=xyz.dtype)
```

シーンで `normalize_object` を使うと、壁までの距離という一番大事な情報が毎フレーム違う倍率で潰れます。ロボット中心のデータは、固定の数で割るだけにします。強度などの追加の列は、0〜1 程度に収まるよう別に割ります。

PointNet++ の `radius` は、この正規化後の単位です。

### 学習時だけのデータ増強

推論では使いません。

```python
def augment(points: np.ndarray, rng: np.random.Generator, jitter: float = 0.01, yaw: bool = False) -> np.ndarray:
    out = points.copy()
    out[:, :3] += rng.normal(0.0, jitter, size=(len(out), 3))
    if yaw:
        a = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(a), np.sin(a)
        out[:, :2] = out[:, :2] @ np.array([[c, -s], [s, c]]).T
    return out
```

ロボットの前向きに意味がある課題（速度指令の回帰など）で z 軸まわりに回すと、点群と正解の指令が食い違います。回すなら、目標位置などのラベルも同じ角度で回します。物体の分類なら、z 軸まわりの回転は安全な増強です。

### まとめて通す

```python
def preprocess(points, rng, *, n=1024, max_range=8.0, z_min=-0.5, z_max=2.5,
               voxel=0.05, z_scale=2.0, train=False):
    points = drop_invalid(points)
    points = crop(points, max_range, z_min, z_max)
    points = voxel_downsample(points, voxel)
    if train:
        points = augment(points, rng)
    points = fix_count(points, n, rng, fps=not train)
    points[:, :3] = normalize_scene(points[:, :3], max_range, z_scale)
    return points.astype(np.float32)
```

最遠点サンプリングを使うと、点群が同じなら毎回ほぼ同じ点が選ばれます。学習中はランダムに選び直したほうが、同じ場面から違う部分集合を見られます。上の例はその使い分けです。

このリポジトリでは同じことを次のように書きます。`rng` を渡すとボクセルの代表点と点数の削減がランダムになり、`augment` を渡すと増強が入ります。推論ではどちらも渡しません。

```python
import json
import numpy as np
from physical_pointnet.preprocess import AugmentConfig, PreprocessConfig, preprocess_scan

cfg = PreprocessConfig.from_dict(json.load(open("configs/preprocess/outdoor.json")))
points = read_frame()  # (N, 4)。x, y, z, intensity

coord, feat, mirrored = preprocess_scan(points, cfg, np.random.default_rng(), AugmentConfig())  # 学習
coord, feat, _ = preprocess_scan(points, cfg)                                                 # 推論
```

`coord` はロボット座標のメートル `(n, 3)` で、エンコーダの近傍探索やボクセル化に使います。`feat` は倍率を掛けた xyz と追加の列 `(n, 3 + E)` で、ネットワークの入力チャネルです。左右反転したとき（`mirrored` が真）は、角速度と目標の左右位置の符号を `dataset.py` が反転します。

## 6. 学習と検証の分け方

連続したフレームは、ほぼ同じ点群です。フレーム単位でランダムに分けると、検証に学習とほぼ同じ絵が入り、成績が実力より良く見えます。系列、走行、日付、場所などの単位で分けます。

```python
def split_by_group(groups: np.ndarray, val_ratio: float, seed: int):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    val = set(rng.permutation(uniq)[: max(1, int(round(len(uniq) * val_ratio)))].tolist())
    is_val = np.array([g in val for g in groups])
    return np.where(~is_val)[0], np.where(is_val)[0]
```

このリポジトリの `physical_pointnet/dataset.py` も、エピソード単位で分けています。

## 7. 保存形式

変換が終わったら、学習で毎回生データを読み直さないように、そろえた配列で保存します。このリポジトリの `npz` は次の列です。

| 列 | 形 | 中身 |
| --- | --- | --- |
| `points` | `(M, N, 3)` | ロボット座標のメートル。正規化前 |
| `status` | `(M, 5)` | スケール済みの状態 |
| `action` | `(M, 2)` | スケール済みの指令 |
| `episode_id` | `(M,)` | 分割の単位 |

正規化は保存後ではなく、学習と推論の直前で同じ関数に通します。割る数（最大距離など）を変えて試すときに、データを作り直さずに済みます。

フレームごとに点数が違う実データでは、`points` を `(N_i, 3 + E)` の配列のリストにします。`DemoDataset` はリストもそのまま受け取ります。1 つの `npz` に入れるならオブジェクト配列になるので、読むときに `allow_pickle=True` が必要です。量が多いときはフレームごとに別ファイルにするほうが扱いやすいです。

## 調べたら残しておくこと

- 列の名前と順番、単位
- 座標系の向きと原点、センサからロボットへの変換
- 1フレームの点数の範囲と、欠損の表し方
- 状態とラベルの時刻の合わせ方と、許した時間差
- 前処理の各値（最大距離、ボクセルの大きさ、点数、割る数）
- 学習と検証を分けた単位

これがあると、別の人が同じデータで同じ入力を再現できます。
