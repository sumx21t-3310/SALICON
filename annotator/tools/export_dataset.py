"""アノテーション結果（SessionResult の JSON）を salicon-vit の学習データ形式に変換する。

出力（salvit/data.py の SaliconDataset が読む構成）::

    out/
      maps/{split}/{image_id}.png        # 連続値の顕著性マップ（0〜255）
      fixations/{split}/{image_id}.png   # 2 値の注視点マップ（注視点の画素が 255）

処理は SALICON 論文（Jiang et al., CVPR 2015）に合わせる。

1. 未完了・サンプルなし・12 Hz 未満の試行を破棄する
2. 静止中の位置を補う（開始時の中央、最初の移動まで、最後の移動から提示終了まで）
3. PCHIP で 100 Hz に再標本化する
4. 試行ごとに速度の大きい半分のサンプル（サッカード）を除く
5. 全被験者の注視点を集計し、ガウシアンフィルターでぼかす（σ は既定で 1 度）

被験者は participantId で区別する（無い場合は JSON ファイルごとに別の被験者とみなす）。
同じ被験者が同じ画像を複数回見ている場合、既定では最初の試行だけを使う（--repeat all で全試行）。

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


HOLD_GAP_MS = 10.0


def gaze_track(trial: dict) -> list[dict]:
    """マウスイベントは動いたときにしか出ないため、静止中の位置を補った軌跡を作る。

    processing.ts の gazeTrack と同じ処理。品質判定には使わない。
    """
    dur = float(trial["durationMs"])
    pts = sorted((p for p in trial["samples"] if 0 <= p["t"] <= dur), key=lambda p: p["t"])
    out: list[dict] = []
    start = trial.get("startGaze")
    if start:
        first = pts[0]["t"] if pts else dur
        if first > 0:
            out.append({"t": 0.0, "x": start["x"], "y": start["y"]})
        if first > HOLD_GAP_MS:
            out.append({"t": first - HOLD_GAP_MS, "x": start["x"], "y": start["y"]})
    out.extend(pts)
    if out and trial.get("completed") and dur - out[-1]["t"] > HOLD_GAP_MS:
        out.append({"t": dur, "x": out[-1]["x"], "y": out[-1]["y"]})
    return out


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
    ap.add_argument("--min-observers", type=int, default=1, help="有効な被験者（重複なし）がこの数未満の画像は出力しない")
    ap.add_argument(
        "--repeat",
        choices=["first", "all"],
        default="first",
        help="同じ被験者が同じ画像を複数回見た場合の扱い。first: 最初の試行だけ使う、all: すべて使う",
    )
    args = ap.parse_args()

    # 画像 ID → 被験者 → 試行（開始時刻順）
    by_image: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    dropped = 0
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        observer = (data.get("participantId") or "").strip() or f"file:{path.resolve()}"
        for tr in data["trials"]:
            if is_valid(tr, args.min_hz):
                by_image[tr["imageId"]][observer].append(tr)
            else:
                dropped += 1

    map_dir = args.out / "maps" / args.split
    fix_dir = args.out / "fixations" / args.split
    map_dir.mkdir(parents=True, exist_ok=True)
    fix_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    repeated = 0
    for image_id, observers in sorted(by_image.items()):
        if len(observers) < args.min_observers:
            print(f"skip {image_id}: 被験者 {len(observers)} 人")
            continue
        trials: list[dict] = []
        for obs_trials in observers.values():
            obs_trials.sort(key=lambda t: t.get("startedAt", ""))
            if args.repeat == "first":
                repeated += len(obs_trials) - 1
                obs_trials = obs_trials[:1]
            trials.extend(obs_trials)
        w, h = trials[0]["imageWidth"], trials[0]["imageHeight"]
        counts = np.zeros((h, w), dtype=np.float64)
        for tr in trials:
            xy = remove_saccades(resample(gaze_track(tr), args.resample_hz), args.keep_ratio)
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

    print(f"{written} 枚を出力しました（破棄した試行: {dropped}、同じ被験者の重複で除いた試行: {repeated}）")


if __name__ == "__main__":
    main()
