# third_party — OSS バックボーン

このディレクトリは PointNet++ / Point Transformer の公開実装をそのまま取り込み、
`physical_pointnet.encoders` の薄いラッパから `import` して使うためのものです。

| パッケージ | 由来 | 用途 |
| --- | --- | --- |
| `pointnet2_pytorch` | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (`pointnet2_utils.py`) | PointNet++ のセット抽象化（MSG / SSG） |
| `point_transformer` | [qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers)（Hengshuang 実装 + `pointnet_util.py`） | Point Transformer（Zhao ほか, ICCV 2021）のエンコーダ |

各サブディレクトリに `LICENSE`（MIT）と取り込み元コミットを記した `SOURCE.txt` があります。

## 想定環境

- Python 3.6
- PyTorch 1.10.0
- CUDA 10.2〜11.3（CPU のみでも、純粋 PyTorch 実装なので動きます）

CUDA 拡張（pointops / spconv など）は使いません。

## 変更点（取り込み時）

- パッケージ相対 import に変更（`from .pointnet_util import ...`）
- Point Transformer 側の `torch.cuda.empty_cache()` 呼び出しを削除（計算結果には影響なし。CPU ビルドでも落ちないようにするため）
- Hydra の `cfg` 依存を外し、`backbone.Backbone` はキーワード引数で構築
