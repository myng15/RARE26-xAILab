"""Architecture: a DINOv2 backbone adapted with LoRA, a patch-token pooling head, and a linear head.

``AttentionPool`` below is our own implementation of the (non-gated) pooling rule of:
    Ilse, M., Tomczak, J., and Welling, M. "Attention-based Deep Multiple Instance Learning." ICML 2018.
    https://arxiv.org/abs/1802.04712 
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

FEAT_DIM = 768
IMG_SIZE = 336


def _dinov2_repo_on_path(dinov2_repo: Path) -> None:
    if str(dinov2_repo) not in sys.path:
        sys.path.insert(0, str(dinov2_repo))


def build_backbone(dinov2_repo: Path, weights: Path | None, img_size: int = IMG_SIZE) -> nn.Module:
    """DINOv2 ViT-B/14 with 4 register tokens, optionally initialized from a checkpoint.

    The challenge submission uses GastroNet-5M self-supervised weights, which are gastrointestinal-domain
    rather than natural-image pretrained. Request access and pass the file via ``weights``.
    """
    _dinov2_repo_on_path(dinov2_repo)
    from dinov2.models import vision_transformer as vits

    backbone = vits.vit_base(img_size=img_size, patch_size=14, init_values=1.0, ffn_layer="mlp",
                             block_chunks=0, num_register_tokens=4,
                             interpolate_antialias=True, interpolate_offset=0.0)
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        state = state.get("teacher", state.get("model", state))
        state = {k.replace("backbone.", ""): v for k, v in state.items() if "head" not in k}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"backbone weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    return backbone


class AttentionPool(nn.Module):
    """Attention-based multiple-instance pooling (Ilse et al., ICML 2018, Eq. 8).

    A small side-network scores every patch token, the scores are softmax-normalized across patches into
    weights, and the frame embedding is their weighted sum. Only image-level labels are used; the
    weighting is learned indirectly from the classification objective.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.score_proj = nn.Linear(feat_dim, hidden_dim)
        self.score_out = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        scores = self.score_out(torch.tanh(self.score_proj(patch_tokens)))
        return (torch.softmax(scores, dim=1) * patch_tokens).sum(dim=1)


class MeanPool(nn.Module):
    """Unweighted mean over patch tokens. No learned parameters."""

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return patch_tokens.mean(dim=1)


POOLINGS = {"attention": AttentionPool, "mean": MeanPool}


class NeoplasiaClassifier(nn.Module):
    """Backbone -> patch-token pooling -> linear head -> one logit per frame."""

    def __init__(self, backbone: nn.Module, pooling: str = "attention", feat_dim: int = FEAT_DIM):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {sorted(POOLINGS)}, got {pooling!r}")
        self.backbone = backbone
        self.pooling_name = pooling
        self.pool = POOLINGS[pooling]() if pooling == "mean" else POOLINGS[pooling](feat_dim)
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.backbone.forward_features(images)["x_norm_patchtokens"]
        return self.head(self.pool(patch_tokens)).squeeze(-1)


def add_lora(model: NeoplasiaClassifier, rank: int = 32, alpha: int = 64,
             dropout: float = 0.1) -> NeoplasiaClassifier:
    """Wrap the backbone's attention qkv projections in LoRA adapters and freeze everything else in it."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=["qkv"], bias="none")
    model.backbone = get_peft_model(model.backbone, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}")
    return model


EXPORT_KEY_MAP = {
    "pool.score_proj.weight": "attn_V.weight",
    "pool.score_proj.bias": "attn_V.bias",
    "pool.score_out.weight": "attn_w.weight",
    "pool.score_out.bias": "attn_w.bias",
    "head.weight": "head.weight",
    "head.bias": "head.bias",
}


def export_head_state(model: NeoplasiaClassifier) -> dict:
    """The pooling and classifier weights, keyed the way the container expects to read them."""
    head_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("backbone."):
            continue
        if key not in EXPORT_KEY_MAP:
            raise KeyError(f"{key!r} has no name in the container's checkpoint format; "
                           f"expected one of {sorted(EXPORT_KEY_MAP)}")
        head_state[EXPORT_KEY_MAP[key]] = value
    return head_state


def export_for_inference(model: NeoplasiaClassifier, path: Path, img_size: int = IMG_SIZE,
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
    """Merge the LoRA adapters into the backbone and save a checkpoint the container can load.

    The exported file has no dependency on the adapter library, so inference needs only the plain
    backbone definition plus the pooling and head weights.
    """
    model.eval()
    merged = model.backbone.merge_and_unload()
    torch.save({"backbone": merged.state_dict(), "head": export_head_state(model),
                "pooling": model.pooling_name, "img_size": img_size,
                "feat_dim": model.head.in_features,
                "mean": list(mean), "std": list(std)}, path)
    print(f"exported {path}")
