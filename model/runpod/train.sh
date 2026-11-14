#!/usr/bin/env bash
# Runpod 上で学習する。Pod が止まっても、同じコマンドで last.pt から再開する。
#
#   bash runpod/train.sh                                  # ViT-B/16（既定）
#   CONFIG=configs/salicon_dinov2_vitb14.yaml bash runpod/train.sh
#   bash runpod/train.sh train.batch_size=32 train.epochs=30   # 設定の上書き
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"

CONFIG="${CONFIG:-configs/salicon_vitb16.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG" .yaml)}"
OUT="$WORKSPACE/runs/$RUN_NAME"
mkdir -p "$OUT"

# SSH が切れても学習が続くよう、ログはファイルにも残す（nohup / tmux と併用推奨）
python scripts/train.py --config "$CONFIG" --resume \
  data.root="$WORKSPACE/data/salicon" \
  output_dir="$OUT" \
  "$@" 2>&1 | tee -a "$OUT/train.log"
