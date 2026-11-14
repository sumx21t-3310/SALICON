#!/usr/bin/env bash
# Runpod の Pod 起動後に 1 回実行する。
# 想定テンプレート: Runpod 公式 PyTorch テンプレート（torch・CUDA 導入済み）
# 永続化したいもの（データ・重み・キャッシュ・学習結果）はすべて /workspace（Network Volume）に置く。
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# torch はテンプレートのものを使い、それ以外だけ入れる（CUDA 版の torch を上書きしないため）
pip install --no-cache-dir timm numpy pillow scipy pyyaml

# timm / Hugging Face の事前学習済み重みを Network Volume にキャッシュする
mkdir -p "$WORKSPACE/.cache/huggingface" "$WORKSPACE/data" "$WORKSPACE/runs"
grep -q HF_HOME ~/.bashrc 2>/dev/null || echo "export HF_HOME=$WORKSPACE/.cache/huggingface" >> ~/.bashrc

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "setup done: $MODEL_DIR"
