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
    return x / (x.sum(1, keepdim=True) + EPS)


def _standardize(x):
    x = _flat(x)
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + EPS)


def kld(pred, gt):
    p, q = _norm_sum(pred), _norm_sum(gt)
    return (q * torch.log(EPS + q / (p + EPS))).sum(1)


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
    """AUC-Judd。注視点の値をしきい値として ROC 曲線下面積を求める。"""
    out = []
    for p, f in zip(_flat(pred), _flat(fix)):
        p = (p - p.min()) / (p.max() - p.min() + EPS)
        thr = p[f > 0]
        n_fix, n_pix = thr.numel(), p.numel()
        if n_fix == 0:
            out.append(torch.tensor(float("nan"), device=p.device))
            continue
        thr, _ = thr.sort(descending=True)
        sorted_all, _ = p.sort()
        # 各しきい値以上の画素数（二分探索）
        above = n_pix - torch.searchsorted(sorted_all, thr, right=False)
        tp = torch.cat([p.new_zeros(1), torch.arange(1, n_fix + 1, device=p.device) / n_fix, p.new_ones(1)])
        fp = torch.cat([p.new_zeros(1), (above - torch.arange(1, n_fix + 1, device=p.device)) / (n_pix - n_fix), p.new_ones(1)])
        out.append(torch.trapz(tp, fp))
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
