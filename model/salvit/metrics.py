"""顕著性評価指標と学習用損失（MIT/Tuebingen ベンチマークの定義に準拠）。

入力はすべて [B,1,H,W]。pred は非負の顕著性マップ（確率マップでも可）。
"""
from __future__ import annotations

import torch

EPS = 1e-7


def _flat(x):
    return x.flatten(1)


def _norm_sum(x):
    x = _flat(x)
    return x / x.sum(1, keepdim=True).clamp_min(EPS)


def _standardize(x):
    # 確率マップ（総和 1）は値が非常に小さいため、std に固定の eps を足すと
    # スケール不変性が崩れる。std が 0 の定数マップだけ 1 で割って 0 にする。
    x = _flat(x)
    std = x.std(1, keepdim=True)
    return (x - x.mean(1, keepdim=True)) / torch.where(std > 0, std, torch.ones_like(std))


def kld(pred, gt):
    # 分母だけに eps を足すと高解像度で偏りが出る（同一分布でも 0 にならない）ため、
    # 確率を下限でクランプしてから対数比を取る。q=0 の画素は寄与 0。
    p, q = _norm_sum(pred), _norm_sum(gt)
    tiny = torch.finfo(p.dtype).tiny
    log_ratio = torch.log(q.clamp_min(tiny)) - torch.log(p.clamp_min(tiny))
    return torch.where(q > 0, q * log_ratio, torch.zeros_like(q)).sum(1)


def cc(pred, gt):
    p, q = _standardize(pred), _standardize(gt)
    return (p * q).mean(1) * p.shape[1] / (p.shape[1] - 1)


def sim(pred, gt):
    p, q = _norm_sum(pred), _norm_sum(gt)
    return torch.minimum(p, q).sum(1)


def nss(pred, fix):
    p, f = _standardize(pred), _flat(fix)
    return (p * f).sum(1) / (f.sum(1) + EPS)


def auc_judd(pred, fix):
    """AUC-Judd。注視点の値をしきい値として ROC 曲線下面積を求める。

    同じ値の注視点が複数ある場合は 1 つのしきい値にまとめ、そのしきい値以上の
    注視点をすべて数える（一様なマップでチャンスレベル 0.5 になる）。
    """
    out = []
    for p, f in zip(_flat(pred), _flat(fix)):
        rng = p.max() - p.min()
        p = (p - p.min()) / (rng if rng > 0 else 1.0)
        fix_vals = p[f > 0]
        n_fix, n_pix = fix_vals.numel(), p.numel()
        if n_fix == 0 or n_fix == n_pix:
            out.append(torch.tensor(float("nan"), device=p.device))
            continue
        fix_sorted, _ = fix_vals.sort()
        all_sorted, _ = p.sort()
        thr = torch.unique(fix_vals).flip(0)  # 降順
        # 各しきい値以上の画素数・注視点数（二分探索）
        n_above_all = n_pix - torch.searchsorted(all_sorted, thr, right=False)
        n_above_fix = n_fix - torch.searchsorted(fix_sorted, thr, right=False)
        tp = n_above_fix.to(p.dtype) / n_fix
        fp = (n_above_all - n_above_fix).to(p.dtype) / (n_pix - n_fix)
        zero, one = p.new_zeros(1), p.new_ones(1)
        out.append(torch.trapz(torch.cat([zero, tp, one]), torch.cat([zero, fp, one])))
    return torch.stack(out)


class SaliencyLoss(torch.nn.Module):
    """KLD + CC + NSS の重み付き和（下がるほど良い）。"""

    def __init__(self, w_kld=1.0, w_cc=0.5, w_nss=0.1):
        super().__init__()
        self.w_kld, self.w_cc, self.w_nss = w_kld, w_cc, w_nss

    def forward(self, prob, gt_map, fix=None):
        loss = self.w_kld * kld(prob, gt_map).mean() - self.w_cc * cc(prob, gt_map).mean()
        if fix is not None and self.w_nss:
            loss = loss - self.w_nss * nss(prob, fix).mean()
        return loss


@torch.no_grad()
def evaluate_batch(pred, gt_map=None, fix=None) -> dict[str, torch.Tensor]:
    res = {}
    if gt_map is not None:
        res["CC"] = cc(pred, gt_map)
        res["KLD"] = kld(pred, gt_map)
        res["SIM"] = sim(pred, gt_map)
    if fix is not None:
        res["NSS"] = nss(pred, fix)
        res["AUC_J"] = auc_judd(pred, fix)
    return res
