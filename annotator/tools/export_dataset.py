"""アノテーション結果（SessionResult の JSON）を salicon-vit の学習データ形式に変換する。

出力（salvit/data.py の SaliconDataset が読む構成）::

    out/
      maps/{split}/{image_id}.png        # 連続値の顕著性マップ（0〜255）
      fixations/{split}/{image_id}.png   # 2 値の注視点マップ（注視点の画素が 255）

処理は SALICON 論文（Jiang et al., CVPR 2015）に合わせる。

1. 未完了・サンプルなし・12 Hz 未満の試行を破棄する
2. PCHIP で 100 Hz に再標本化する
3. 試行ごとに速度の大きい半分のサンプル（サッカード）を除く
4. 全被験者の注視点を集計し、ガウシアンフィルターでぼかす（σ は既定で 1 度）

使い方::

    python tools/export_dataset.py results/*.json --out data/my_ui_dataset --split train
    # 画像も一緒にコピーする場合
    python tools/export_dataset.py results/*.json --out data/my_ui_dataset --split train --images-dir raw_images
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter


def resample(samples: list[dict], hz: float = 100.0) -> np.ndarray:
    pts = sorted(samples, key=lambda s: s["t"])
    t = np.array([p["t"] for p in pts], dtype=np.float64)
    xy = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
    t, idx = np.unique(t, return_index=True)
    xy = xy[idx]
    if len(t) < 2:
        return xy
    ti = np.arange(t[0], t[-1] + 1e-9, 1000.0 / hz)
    return np.stack([PchipInterpolator(t, xy[:, k])(ti) for k in range(2)], axis=1)


def remove_saccades(xy: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    if len(xy) < 3:
        return xy
    speed = np.linalg.norm(np.gradient(xy, axis=0), axis=1)
    keep = np.argsort(speed, kind="stable")[: int(round(len(xy) * keep_ratio))]
    return xy[np.sort(keep)]


def is_valid(trial: dict, min_hz: float) -> bool:
    return bool(trial.get("completed")) and len(trial["samples"]) > 0 and trial["quality"]["samplingRateHz"] >= min_hz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path, help="SessionResult の JSON（複数可）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--images-dir", type=Path, help="元画像のフォルダー。指定すると images/{split} にコピーする")
    ap.add_argument("--sigma-deg", type=float, default=1.0, help="ガウシアンの σ（度）。論文に明記がないため既定は 1 度")
    ap.add_argument("--resample-hz", type=float, default=100.0)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--min-hz", type=float, default=12.0)
    ap.add_argument("--min-observers", type=int, default=1, help="有効な被験者がこの数未満の画像は出力しない")
    args = ap.parse_args()

    by_image: dict[str, list[dict]] = defaultdict(list)
    dropped = 0
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        for tr in data["trials"]:
            if is_valid(tr, args.min_hz):
                by_image[tr["imageId"]].append(tr)
            else:
                dropped += 1

    map_dir = args.out / "maps" / args.split
    fix_dir = args.out / "fixations" / args.split
    map_dir.mkdir(parents=True, exist_ok=True)
    fix_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id, trials in sorted(by_image.items()):
        if len(trials) < args.min_observers:
            print(f"skip {image_id}: 被験者 {len(trials)} 人")
            continue
        w, h = trials[0]["imageWidth"], trials[0]["imageHeight"]
        counts = np.zeros((h, w), dtype=np.float64)
        for tr in trials:
            xy = remove_saccades(resample(tr["samples"], args.resample_hz), args.keep_ratio)
            xs = np.floor(xy[:, 0] * w / tr["imageWidth"]).astype(int)
            ys = np.floor(xy[:, 1] * h / tr["imageHeight"]).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            np.add.at(counts, (ys[ok], xs[ok]), 1)
        ppd = trials[0]["foveation"]["pixelsPerDegree"]
        sal = gaussian_filter(counts, sigma=args.sigma_deg * ppd, mode="constant")
        if sal.max() > 0:
            sal /= sal.max()
        Image.fromarray((sal * 255).round().astype(np.uint8)).save(map_dir / f"{image_id}.png")
        Image.fromarray(((counts > 0) * 255).astype(np.uint8)).save(fix_dir / f"{image_id}.png")

        if args.images_dir:
            src = next((p for p in args.images_dir.glob(f"{image_id}.*")), None)
            if src:
                dst = args.out / "images" / args.split
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / src.name)
            else:
                print(f"warn: 画像が見つかりません: {image_id}")
        written += 1

    print(f"{written} 枚を出力しました（破棄した試行: {dropped}）")


if __name__ == "__main__":
    main()
