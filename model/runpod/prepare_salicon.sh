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
  inner="$(dirname "$(find "$tmp" -type d -name train | head -n1)")"
  mkdir -p "$DST/$name"
  mv "$inner"/* "$DST/$name/"
  rm -rf "$tmp"
  echo "ok: $name"
done

for d in images/train images/val maps/train maps/val fixations/train fixations/val; do
  printf "%-18s %6s files\n" "$d" "$(ls "$DST/$d" 2>/dev/null | wc -l)"
done
