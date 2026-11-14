#!/usr/bin/env bash
# SALICON の zip を展開し、model が読む構成に並べる。
#
# 使い方:
#   1. http://salicon.net/challenge-2017/ から images.zip / maps.zip / fixations.zip を取得し、
#      $WORKSPACE/data/salicon_zip/ に置く（runpodctl send/receive、または Pod 上で直接ダウンロード）
#   2. bash runpod/prepare_salicon.sh
#
# 展開後: $WORKSPACE/data/salicon/{images,maps,fixations}/{train,val[,test]}/
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SRC="${1:-$WORKSPACE/data/salicon_zip}"
DST="${2:-$WORKSPACE/data/salicon}"
mkdir -p "$DST"

for name in images maps fixations; do
  zip="$SRC/$name.zip"
  if [[ ! -f "$zip" ]]; then
    echo "skip: $zip が見つかりません"; continue
  fi
  if [[ -d "$DST/$name/train" ]]; then
    echo "skip: $DST/$name は展開済み"; continue
  fi
  tmp="$(mktemp -d "$DST/.tmp_$name.XXXX")"
  unzip -q "$zip" -d "$tmp"
  # zip によって最上位フォルダの有無が違うため、train/ を含む階層を探して移動する
  train_dir="$(find "$tmp" -type d -name train -print -quit)"
  if [[ -z "$train_dir" ]]; then
    echo "error: $zip に train/ ディレクトリがありません（展開を中止）" >&2
    rm -rf "$tmp"; exit 1
  fi
  inner="$(dirname "$train_dir")"
  case "$inner" in "$tmp"|"$tmp"/*) ;; *) echo "error: 想定外の展開先 $inner" >&2; rm -rf "$tmp"; exit 1;; esac
  mkdir -p "$DST/$name"
  mv "$inner"/* "$DST/$name/"
  rm -rf "$tmp"
  echo "ok: $name"
done

for d in images/train images/val maps/train maps/val fixations/train fixations/val; do
  printf "%-18s %6s files\n" "$d" "$(ls "$DST/$d" 2>/dev/null | wc -l)"
done
