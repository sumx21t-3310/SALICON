"""検証データで CC / KLD / SIM / NSS / AUC-Judd を計算する。

例: python scripts/evaluate.py --ckpt runs/salicon_vitb16/best.pt --root data/salicon --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salvit.data import SaliconDataset, collate  # noqa: E402
from salvit.engine import evaluate  # noqa: E402
from salvit.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg["model"]["pretrained"] = False
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device)
    ds = SaliconDataset(args.root, args.split, cfg["data"]["img_size"], limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
    print(json.dumps(evaluate(model, dl, device, amp=True), indent=2))


if __name__ == "__main__":
    main()
