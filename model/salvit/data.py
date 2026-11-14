"""SALICON データセットの読み込み。

想定するディレクトリ構成（SALICON 2017 / LSUN 配布版）::

    root/
      images/{train,val,test}/COCO_*.jpg     # 640x480
      maps/{train,val}/COCO_*.png            # 連続値の顕著性マップ（正解）
      fixations/{train,val}/COCO_*.mat       # 被験者ごとの注視点（NSS・AUC 用）

fixations が無い場合は、注視点を必要とする指標（NSS、AUC）をスキップする。

UI・イラストなど独自データでファインチューニングする場合も同じ構成で置けばよい。
画像のファイル名・縦横比は自由で、注視点は .mat の代わりに 2 値 PNG
（注視点の画素が 0 以外）でも渡せる。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

ORIG_H, ORIG_W = 480, 640


def load_fixation_map(mat_path: Path, h: int = ORIG_H, w: int = ORIG_W) -> np.ndarray:
    """SALICON の .mat から 2 値の注視点マップ (h, w) を作る。"""
    from scipy.io import loadmat

    mat = loadmat(str(mat_path))
    if "resolution" in mat:
        h, w = (int(v) for v in np.asarray(mat["resolution"]).ravel()[:2])
    fix = np.zeros((h, w), dtype=np.uint8)
    gaze = mat["gaze"].ravel()
    for subj in gaze:
        pts = np.asarray(subj["fixations"]).reshape(-1, 2)  # [x(col), y(row)], 1 始まり
        if pts.size == 0:
            continue
        xs = np.clip(pts[:, 0].astype(int) - 1, 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int) - 1, 0, h - 1)
        fix[ys, xs] = 1
    return fix


class SaliconDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size=(384, 512),
        augment: bool = False,
        use_fixations: bool = True,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = tuple(img_size)
        self.augment = augment
        img_dir = self.root / "images" / split
        self.images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if limit:
            self.images = self.images[:limit]
        if not self.images:
            raise FileNotFoundError(f"画像が見つかりません: {img_dir}")
        self.map_dir = self.root / "maps" / split
        self.fix_dir = self.root / "fixations" / split
        self.has_maps = self.map_dir.is_dir()
        self.has_fix = use_fixations and self.fix_dir.is_dir()

    def __len__(self):
        return len(self.images)

    def _load_fix(self, stem: str) -> np.ndarray:
        mat = self.fix_dir / f"{stem}.mat"
        if mat.exists():
            return load_fixation_map(mat)
        png = self.fix_dir / f"{stem}.png"
        return (np.asarray(Image.open(png).convert("L")) > 0).astype(np.uint8)

    def _resize(self, t: torch.Tensor, mode: str) -> torch.Tensor:
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(t[None], size=self.img_size, mode=mode, **kw)[0]

    def __getitem__(self, idx):
        path = self.images[idx]
        img = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        orig_hw = img.shape[-2:]
        sample = {"name": path.stem, "orig_size": torch.tensor(orig_hw)}

        flip = self.augment and random.random() < 0.5
        if self.augment:
            # 顕著性は色に依存するため、幾何変換は左右反転のみ・色の変化は弱めにする
            img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
            img = img.clamp(0, 1)
        img = self._resize(img, "bilinear")
        if flip:
            img = img.flip(-1)
        sample["image"] = img

        if self.has_maps:
            m = np.asarray(Image.open(self.map_dir / f"{path.stem}.png").convert("L"), dtype=np.float32) / 255.0
            m = torch.from_numpy(m)[None]
            sample["map_full"] = m.flip(-1) if flip else m  # 評価用（元解像度）
            m = self._resize(m, "bilinear")
            sample["map"] = m.flip(-1) if flip else m
        if self.has_fix:
            fx = torch.from_numpy(self._load_fix(path.stem))[None].float()
            sample["fix_full"] = fx.flip(-1) if flip else fx
            # 学習用は最大値プーリングで縮小し、注視点が消えないようにする
            fs = F.adaptive_max_pool2d(fx[None], self.img_size)[0]
            sample["fix"] = fs.flip(-1) if flip else fs
        return sample


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k == "name":
            out[k] = vals
        elif k.endswith("_full"):
            # 元解像度は画像によって異なる可能性があるため、そろっていれば stack、違えば list
            shapes = {v.shape for v in vals}
            out[k] = torch.stack(vals) if len(shapes) == 1 else vals
        else:
            out[k] = torch.stack(vals)
    return out
