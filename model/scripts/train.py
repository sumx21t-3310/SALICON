"""学習スクリプト。SALICON での学習と、UI・イラストデータへのファインチューニングを兼ねる。

例:
    python scripts/train.py --config configs/salicon_vitb16.yaml
    python scripts/train.py --config configs/finetune_ui.yaml init_from=runs/salicon_vitb16/best.pt
    python scripts/train.py --config configs/salicon_vitb16.yaml --resume   # output_dir/last.pt から再開
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salvit.data import SaliconDataset, collate  # noqa: E402
from salvit.engine import evaluate  # noqa: E402
from salvit.metrics import SaliencyLoss  # noqa: E402
from salvit.model import build_model, logits_to_prob  # noqa: E402
from salvit.utils import cosine_with_warmup, load_config, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="output_dir/last.pt があれば続きから学習する")
    ap.add_argument("overrides", nargs="*", help="key.sub=value 形式で設定を上書き")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg, tcfg = cfg["data"], cfg["train"]
    out_dir = Path(cfg.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    train_ds = SaliconDataset(dcfg["root"], dcfg.get("train_split", "train"), dcfg["img_size"], augment=True,
                              limit=dcfg.get("limit_train"))
    val_ds = SaliconDataset(dcfg["root"], dcfg.get("val_split", "val"), dcfg["img_size"], augment=False,
                            limit=dcfg.get("limit_val"))
    kw = dict(num_workers=dcfg.get("num_workers", 4), collate_fn=collate, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=tcfg.get("eval_batch_size", tcfg["batch_size"]), **kw)

    model = build_model(cfg)
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd.get("model", sd), strict=False)
        print(f"init_from={cfg['init_from']} missing={len(missing)} unexpected={len(unexpected)}")
        model.freeze_blocks(cfg["model"].get("freeze_blocks", 0))
    model.to(device)

    opt = torch.optim.AdamW(model.param_groups(tcfg["lr"], tcfg.get("backbone_lr_mult", 0.1),
                                               tcfg.get("weight_decay", 0.05)))
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(train_dl) / accum)  # 端数の勾配もエポック末に 1 回で更新する
    total = tcfg["epochs"] * steps_per_epoch
    sched = cosine_with_warmup(opt, int(total * tcfg.get("warmup_ratio", 0.05)), total)
    lw = cfg.get("loss", {})
    criterion = SaliencyLoss(lw.get("kld", 1.0), lw.get("cc", 0.5), lw.get("nss", 0.1))
    amp = tcfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    monitor = tcfg.get("monitor", "CC")
    best = -float("inf")
    start_epoch = 0
    if args.resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        print(f"resume: epoch {start_epoch} から再開（best {monitor}={best:.4f}）")

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, run = time.time(), 0.0
        n_batches = len(train_dl)
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_dl):
            # 最後の端数グループは、その実バッチ数で割って平均をそろえる
            group_start = it // accum * accum
            group_size = min(accum, n_batches - group_start)
            img = batch["image"].to(device, non_blocking=True)
            gt = batch["map"].to(device, non_blocking=True)
            fix = batch["fix"].to(device, non_blocking=True) if "fix" in batch else None
            with torch.autocast(device.type, enabled=amp):
                logits = model(img)
            loss = criterion(logits_to_prob(logits.float()), gt, fix) / group_size
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0 or it + 1 == n_batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("clip_grad", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += loss.item() * group_size
            if (it + 1) % tcfg.get("log_every", 50) == 0:
                print(f"ep {epoch} it {it + 1}/{len(train_dl)} loss {run / (it + 1):.4f}", flush=True)

        metrics = evaluate(model, val_dl, device, amp)
        score = metrics.get(monitor, float("nan"))
        if monitor == "KLD":
            score = -score
        log = {"epoch": epoch, "train_loss": run / max(1, len(train_dl)), "time": time.time() - t0, **metrics}
        print(json.dumps(log), flush=True)
        with open(out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        is_best = score > best
        if is_best:
            best = score
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics, "best": best}
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")  # 推論用（最適化状態を含めず軽量）
        ckpt.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict())
        torch.save(ckpt, out_dir / "last.tmp")
        (out_dir / "last.tmp").replace(out_dir / "last.pt")  # 途中停止でも壊れないよう置き換えで保存
        if is_best:
            print(f"best {monitor} 更新: {metrics.get(monitor):.4f}")


if __name__ == "__main__":
    main()
