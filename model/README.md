# model

Vision Transformer をバックボーンにした、SALICON 向けの顕著性（視線）予測ベースラインです。
SALICON で学習したモデルを、UI 画面やキャラクターイラストのデータでファインチューニングし、視線設計の分析に使うことを想定しています。

## モデル構成

- エンコーダ: timm の ViT（既定は `vit_base_patch16_384.augreg_in21k_ft_in1k`、DINOv2 版の設定も同梱）
- デコーダ: DPT 風。ViT の中間 4 層を 1/4〜1/32 の解像度に再構成し、上位から順に融合
- 出力: 入力と同じ解像度のロジット。空間方向の softmax で確率マップにする
- 位置バイアス: 学習可能なガウス事前分布。中心位置も学習するため、写真の中心バイアスにも UI の左上バイアスにも合わせられる
- 損失: KLD + CC + NSS の重み付き和（`loss` で変更可能）
- 評価指標: CC、KLD、SIM、NSS、AUC-Judd（元解像度で計算）

UI・イラスト向けに次の点を考慮しています。

- 入力の縦横比は自由です。パッチサイズで割り切れない場合は内部でパディングして切り戻します。
- 位置埋め込みの基準サイズ（`pos_embed_size`）を入力サイズと分けているため、縦長のスマホ UI など別サイズのデータでもそのまま重みを引き継げます。
- `freeze_blocks` で前半ブロックを凍結でき、小規模データでの過学習を抑えられます。
- `predict.py` は領域（ボタン、顔など）ごとの注視確率の割合を CSV で出力できます。

## 準備

```bash
pip install -r requirements.txt
```

SALICON（http://salicon.net/ 、LSUN 2017 版）を次の構成で置きます。

```
data/salicon/
  images/{train,val,test}/COCO_*.jpg
  maps/{train,val}/COCO_*.png
  fixations/{train,val}/COCO_*.mat
```

独自データ（UI・イラスト）も同じ構成で置きます。`annotator/tools/export_dataset.py` の出力はこの形式です。ファイル名・縦横比は自由で、注視点は `.mat` の代わりに 2 値 PNG でも構いません。`fixations` が無い場合、NSS と AUC は計算しません。

## 使い方

```bash
# 1. SALICON で学習
python scripts/train.py --config configs/salicon_vitb16.yaml
#    DINOv2 版: configs/salicon_dinov2_vitb14.yaml
#    設定は key=value で上書き可能（例: train.batch_size=8 data.root=/path/to/salicon）

# 2. 評価
python scripts/evaluate.py --ckpt runs/salicon_vitb16/best.pt --root data/salicon --split val

# 3. UI・イラストへのファインチューニング
python scripts/train.py --config configs/finetune_ui.yaml init_from=runs/salicon_vitb16/best.pt

# 4. 任意の画像で推論（ヒートマップと領域ごとの注視割合）
python scripts/predict.py --ckpt runs/finetune_ui/best.pt --input screenshots/ --out pred/ --regions regions.json
```

`regions.json` の例（座標は元画像のピクセル）:

```json
{"screen01.png": {"logo": [0, 0, 200, 80], "cta_button": [500, 600, 700, 660]}}
```

`regions.csv` の `lift` は「注視割合 ÷ 面積割合」で、1 を超える領域は面積の割に見られやすいことを示します。

## Runpod での学習

Runpod 公式の PyTorch テンプレートと Network Volume（`/workspace` にマウント）を前提にしています。データ・事前学習済み重みのキャッシュ・チェックポイントはすべて `/workspace` に置くため、Pod を止めても消えません。

```bash
cd /workspace && git clone https://github.com/sumx21t-3310/SALICON-.git && cd SALICON-/model
bash runpod/setup.sh                  # timm などを導入（torch はテンプレートのものを使用）

# SALICON の images.zip / maps.zip / fixations.zip を /workspace/data/salicon_zip/ に置いてから
bash runpod/prepare_salicon.sh        # /workspace/data/salicon/ に展開

tmux new -s train                     # SSH が切れても学習を続けるため
bash runpod/train.sh                  # 結果は /workspace/runs/salicon_vitb16/
```

- Pod が止まった場合も、同じ `bash runpod/train.sh` で `last.pt` から再開します。
- 別の設定: `CONFIG=configs/salicon_dinov2_vitb14.yaml bash runpod/train.sh`
- 設定の上書き: `bash runpod/train.sh train.batch_size=32`
- GPU の目安: ViT-B/16・384x512・バッチ 16 で VRAM 約 16〜20GB（未計測の見積もり）（RTX 4090 / A5000 / L4 以上）。VRAM が足りない場合は `train.batch_size=8 train.grad_accum=2` にします。
- 手元へ重みを持ち帰る: `runpodctl send /workspace/runs/salicon_vitb16/best.pt`

## 動作確認

```bash
python tests/make_dummy_salicon.py /tmp/dummy_salicon
python scripts/train.py --config configs/salicon_vitb16.yaml data.root=/tmp/dummy_salicon \
  model.backbone=vit_tiny_patch16_224 model.pretrained=false train.epochs=3 train.batch_size=4 data.num_workers=0
```

## 目安

ViT-B/16 のパラメータ数は約 99M です。SALICON 全体（学習 10,000 枚）を 384x512・20 エポックで学習する場合、GPU 1 枚（VRAM 16GB 以上）を想定しています。
