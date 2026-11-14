# SALICON

Vision Transformer ベースの顕著性予測モデルと、その学習データを集めるアノテーションツールのモノレポです。ファインチューニング後のモデルは、UI やキャラクターイラストの視線設計の分析に使います。

| ディレクトリ | 内容 |
| --- | --- |
| [`annotator/`](annotator/) | SALICON（Jiang et al., CVPR 2015）方式のマウス追跡で注視データを集めるデスクトップアプリ（Tauri + React） |
| [`model/`](model/) | SALICON 用 ViT 顕著性予測モデル（学習・評価・推論） |

## データの流れ

1. `annotator` で UI 画像やイラストを提示し、参加者ごとの注視データ（JSON）を集める
2. `annotator/tools/export_dataset.py` で、`model` の学習データ形式（`maps/`、`fixations/`）に変換する
3. `model` で SALICON 事前学習済みの重みからファインチューニングする

## 学習（Runpod）

モデルの学習は Runpod を想定しています。手順は [`model/README.md`](model/README.md#runpod-での学習) を参照してください。
