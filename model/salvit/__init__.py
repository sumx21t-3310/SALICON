"""SALICON 向け ViT 顕著性予測ベースライン。"""
from .model import SaliencyViT, build_model

__all__ = ["SaliencyViT", "build_model"]
