from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from .metrics import evaluate_batch
from .model import logits_to_prob


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False) -> dict[str, float]:
    """元解像度（SALICON は 640x480）へ戻してから指標を計算する。"""
    model.eval()
    sums, counts = defaultdict(float), defaultdict(int)
    for batch in loader:
        img = batch["image"].to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(img)
        prob = logits_to_prob(logits.float())
        for i in range(img.shape[0]):
            gt = batch["map_full"][i] if "map_full" in batch else None
            fx = batch["fix_full"][i] if "fix_full" in batch else None
            size = (gt if gt is not None else fx).shape[-2:] if (gt is not None or fx is not None) else prob.shape[-2:]
            p = F.interpolate(prob[i : i + 1], size=size, mode="bilinear", align_corners=False)
            res = evaluate_batch(
                p,
                gt[None].to(device) if gt is not None else None,
                fx[None].to(device) if fx is not None else None,
            )
            for k, v in res.items():
                v = v.item()
                if v == v:  # NaN を除外
                    sums[k] += v
                    counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}
