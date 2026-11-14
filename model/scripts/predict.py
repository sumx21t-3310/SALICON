"""任意の画像（UI のスクリーンショット、キャラクターイラストなど）の顕著性マップを出力する。

縦横比を保ったまま長辺を --long-side にそろえて推論し、元解像度へ戻して保存する。

出力:
    <out>/<name>_map.png       顕著性マップ（0-255、グレースケール）
    <out>/<name>_overlay.png   元画像へのヒートマップ重ね合わせ
    <out>/regions.csv          --regions を指定した場合、領域ごとの注視確率の割合

--regions の JSON 形式（座標は元画像のピクセル、[x0, y0, x1, y1]）:
    {"screen01.png": {"logo": [0, 0, 200, 80], "cta_button": [500, 600, 700, 660]}}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salvit.model import build_model, logits_to_prob  # noqa: E402


def jet(x: np.ndarray) -> np.ndarray:
    """0-1 の配列を jet 風カラーマップの RGB (uint8) にする。matplotlib 不要。"""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = int(sigma * 3) * 2 + 1
    t = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(t**2) / (2 * sigma**2))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


@torch.no_grad()
def predict_one(model, img: Image.Image, long_side: int, device, blur_sigma: float) -> np.ndarray:
    w, h = img.size
    s = long_side / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    x = torch.from_numpy(np.asarray(img.convert("RGB").resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255)
    x = x.permute(2, 0, 1)[None].to(device)
    prob = logits_to_prob(model(x).float())
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    prob = gaussian_blur(prob, blur_sigma * max(w, h) / 640)
    prob = prob / prob.sum()
    return prob[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="画像ファイルまたはフォルダ")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--blur-sigma", type=float, default=0.0, help="640px 換算のガウスぼかし σ")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--regions", help="領域ごとの注視割合を求める JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg["model"]["pretrained"] = False
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    inp = Path(args.input)
    files = sorted(p for p in inp.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if inp.is_dir() else [inp]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions = json.loads(Path(args.regions).read_text()) if args.regions else {}
    rows = []

    for f in files:
        img = Image.open(f)
        prob = predict_one(model, img, args.long_side, device, args.blur_sigma)
        vis = prob / (prob.max() + 1e-12)
        Image.fromarray((vis * 255).astype(np.uint8)).save(out / f"{f.stem}_map.png")
        base = np.asarray(img.convert("RGB"), dtype=np.float32)
        heat = jet(vis).astype(np.float32)
        a = (args.alpha * np.clip(vis * 1.5, 0, 1))[..., None]
        Image.fromarray((base * (1 - a) + heat * a).astype(np.uint8)).save(out / f"{f.stem}_overlay.png")
        for rname, (x0, y0, x1, y1) in regions.get(f.name, {}).items():
            area = (x1 - x0) * (y1 - y0) / prob.size
            share = float(prob[y0:y1, x0:x1].sum())
            rows.append({"image": f.name, "region": rname, "attention_share": round(share, 4),
                         "area_share": round(area, 4), "lift": round(share / max(area, 1e-9), 3)})
        print(f"saved {f.stem}")

    if rows:
        with open(out / "regions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"regions.csv: {len(rows)} 行")


if __name__ == "__main__":
    main()
