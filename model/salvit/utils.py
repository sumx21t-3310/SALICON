from __future__ import annotations

import math
import random

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        from pathlib import Path

        parent = load_config(str(Path(path).parent / base))
        cfg = deep_update(parent, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        d = cfg
        *ks, last = key.split(".")
        for k in ks:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


def deep_update(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
