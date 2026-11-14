"""ViT エンコーダ + DPT 風マルチスケールデコーダによる顕著性予測モデル。

ViT の中間層 4 つからトークンを取り出し、解像度の異なる特徴マップ
(1/4, 1/8, 1/16, 1/32) に再構成してから、上位層から順に融合して
入力解像度の顕著性マップ（ロジット）を出力する。
"""
from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return x + self.block(x)


class FusionBlock(nn.Module):
    """上位層の特徴と現在の層の特徴を足し合わせ、2 倍にアップサンプルする。"""

    def __init__(self, ch: int):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(ch)
        self.rcu_out = ResidualConvUnit(ch)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


class Reassemble(nn.Module):
    """ViT の単一解像度トークンマップを指定の倍率へ変換する。"""

    def __init__(self, in_ch: int, out_ch: int, scale: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 1)]
        if scale == 4:
            layers.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=4))
        elif scale == 2:
            layers.append(nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2))
        elif scale == 0.5:
            layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        elif scale != 1:
            raise ValueError(f"unsupported scale {scale}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SaliencyViT(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_base_patch16_384.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        img_size: Sequence[int] = (384, 512),
        out_indices: Sequence[int] | None = None,
        decoder_ch: int = 256,
        drop_path_rate: float = 0.1,
        center_bias: bool = True,
        freeze_blocks: int = 0,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            img_size=self.img_size,
            dynamic_img_size=True,
            drop_path_rate=drop_path_rate,
        )
        depth = len(self.encoder.blocks)
        if out_indices is None:
            # 12 層なら [2, 5, 8, 11] のように均等に 4 層を取り出す
            out_indices = [depth // 4 * (i + 1) - 1 for i in range(4)]
        self.out_indices = list(out_indices)
        embed_dim = self.encoder.embed_dim

        # patch 16 の場合 1/16 解像度 → (x4, x2, x1, x0.5) で 1/4, 1/8, 1/16, 1/32
        self.reassemble = nn.ModuleList(
            [Reassemble(embed_dim, decoder_ch, s) for s in (4, 2, 1, 0.5)]
        )
        self.fusion = nn.ModuleList([FusionBlock(decoder_ch) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        # 学習可能な位置バイアス（ガウス事前分布の対数をロジットに加算）。
        # 写真は中心に、UI は左上に注視が偏るため、中心位置 mu も学習させる。
        self.center_bias = center_bias
        if center_bias:
            self.cb_mu = nn.Parameter(torch.zeros(2))  # (y, x)、-1〜1 の正規化座標
            self.cb_log_sigma = nn.Parameter(torch.tensor([-1.0, -1.0]))  # (y, x)
            self.cb_weight = nn.Parameter(torch.tensor(1.0))
        self.freeze_blocks(freeze_blocks)

        data_cfg = timm.data.resolve_data_config({}, model=self.encoder)
        self.register_buffer("mean", torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(data_cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def _center_prior(self, h: int, w: int, device, dtype):
        ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        my, mx = self.cb_mu.to(dtype)
        sy, sx = self.cb_log_sigma.exp().to(dtype)
        g = -((ys[:, None] - my) ** 2) / (2 * sy**2) - ((xs[None, :] - mx) ** 2) / (2 * sx**2)
        return (self.cb_weight.to(dtype) * g)[None, None]

    def freeze_blocks(self, n: int):
        """パッチ埋め込みと先頭 n ブロックを凍結する（小規模な UI・イラストデータでの過学習対策）。

        n < 0 でエンコーダ全体を凍結する。
        """
        if n == 0:
            return
        blocks = self.encoder.blocks
        n = len(blocks) if n < 0 else min(n, len(blocks))
        frozen = [self.encoder.patch_embed, *blocks[:n]]
        for name in ("cls_token", "reg_token", "pos_embed"):
            t = getattr(self.encoder, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad_(False)
        for mod in frozen:
            for p in mod.parameters():
                p.requires_grad_(False)

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1, weight_decay: float = 0.05):
        enc, dec = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else dec).append(p)
        return [
            {"params": enc, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": dec, "lr": lr, "weight_decay": weight_decay},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W]、値域 0-1 の RGB。戻り値: [B,1,H,W] のロジット。"""
        H, W = x.shape[-2:]
        x = (x - self.mean) / self.std
        # パッチサイズで割り切れない縦横比（UI 画面など）は右下を反射パディングし、最後に切り戻す
        ph, pw = self.encoder.patch_embed.patch_size
        pad_h, pad_w = (-H) % ph, (-W) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        feats = self.encoder.forward_intermediates(
            x, indices=self.out_indices, intermediates_only=True, output_fmt="NCHW"
        )
        feats = [r(f) for r, f in zip(self.reassemble, feats)]
        out = self.fusion[3](feats[3])
        out = self.fusion[2](out, feats[2])
        out = self.fusion[1](out, feats[1])
        out = self.fusion[0](out, feats[0])
        out = self.head(out)
        out = F.interpolate(out, size=(H + pad_h, W + pad_w), mode="bilinear", align_corners=False)
        out = out[..., :H, :W]
        if self.center_bias:
            out = out + self._center_prior(H, W, out.device, out.dtype)
        return out


def build_model(cfg: dict) -> SaliencyViT:
    m = cfg.get("model", {})
    return SaliencyViT(
        backbone=m.get("backbone", "vit_base_patch16_384.augreg_in21k_ft_in1k"),
        pretrained=m.get("pretrained", True),
        # 位置埋め込みの基準サイズ。入力サイズとは独立に固定し、別サイズのデータで
        # ファインチューニングしても重みの形が変わらないようにする（推論時は補間される）
        img_size=m.get("pos_embed_size", (384, 512)),
        out_indices=m.get("out_indices"),
        decoder_ch=m.get("decoder_ch", 256),
        drop_path_rate=m.get("drop_path_rate", 0.1),
        center_bias=m.get("center_bias", True),
        freeze_blocks=m.get("freeze_blocks", 0),
    )


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    """空間方向の softmax で総和 1 の確率マップへ変換する。"""
    b, c, h, w = logits.shape
    return F.softmax(logits.flatten(1), dim=1).view(b, c, h, w)
