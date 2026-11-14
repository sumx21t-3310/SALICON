"""動作確認用に SALICON と同じ構成のダミーデータを作る。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dummy_salicon")
rng = np.random.default_rng(0)
H, W = 480, 640
yy, xx = np.mgrid[0:H, 0:W]
for split, n in [("train", 8), ("val", 4), ("test", 2)]:
    for d in ["images"] + (["maps", "fixations"] if split != "test" else []):
        (root / d / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name = f"COCO_{split}2014_{i:012d}"
        img = (rng.random((H, W, 3)) * 60).astype(np.uint8)
        cy, cx = rng.integers(80, H - 80), rng.integers(80, W - 80)
        img[cy - 40 : cy + 40, cx - 40 : cx + 40] = [240, 40, 40]  # 目立つ赤い四角
        Image.fromarray(img).save(root / "images" / split / f"{name}.jpg")
        if split == "test":
            continue
        m = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 40.0**2))
        Image.fromarray((m / m.max() * 255).astype(np.uint8)).save(root / "maps" / split / f"{name}.png")
        gaze = np.zeros((5,), dtype=[("location", "O"), ("timestamp", "O"), ("fixations", "O")])
        for s in range(5):
            pts = np.stack([rng.normal(cx, 25, 6), rng.normal(cy, 25, 6)], 1).clip([1, 1], [W, H]).astype(int)
            gaze[s] = (pts, np.arange(6), pts)
        savemat(root / "fixations" / split / f"{name}.mat",
                {"image": name, "resolution": np.array([H, W]), "gaze": gaze.reshape(-1, 1)})
print(f"dummy data -> {root}")
