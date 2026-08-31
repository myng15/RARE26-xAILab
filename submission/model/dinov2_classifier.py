"""Per-frame DINOv2 ensemble classifier with attention-based patch pooling.

Loads N checkpoints, each a {backbone, head} state dict with LoRA merged into a plain DINOv2 ViT-B/14+reg
backbone. For each frame, the backbone's patch tokens are combined by a small attention layer into a single
embedding, which a linear head maps to a logit. Members' sigmoid outputs are averaged into one likelihood.
Each frame's likelihood depends only on that frame, so the output is invariant to input batching.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# The DINOv2 code lives beside this package in the image (see Dockerfile).
_DINOV2_REPO = Path(__file__).resolve().parent.parent / "third_party" / "dinov2"
if str(_DINOV2_REPO) not in sys.path:
    sys.path.insert(0, str(_DINOV2_REPO))


def _build_dinov2_backbone(img_size: int):
    """DINOv2 ViT-B/14 with 4 register tokens. LoRA-merged weights load into this."""
    from dinov2.models import vision_transformer as vits
    return vits.vit_base(img_size=img_size, patch_size=14, init_values=1.0, ffn_layer="mlp",
                         block_chunks=0, num_register_tokens=4,
                         interpolate_antialias=True, interpolate_offset=0.0)


class AttentionPool(nn.Module):
    """Attention-based multiple-instance pooling over patch tokens (Ilse et al., ICML 2018, Eq. 8): a
    small side-network scores each patch, the scores are softmax-normalised across patches into weights,
    and the frame embedding is their weighted sum."""

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score_proj = nn.Linear(feat_dim, hidden_dim)
        self.score_out = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, N, D) -> pooled: (B, D)
        scores = self.score_out(torch.tanh(self.score_proj(patch_tokens)))   # (B, N, 1)
        weights = torch.softmax(scores, dim=1)
        return (weights * patch_tokens).sum(dim=1)


class DINOv2AttentionEnsemble:
    def __init__(self, checkpoint_paths, device=None, batch_size=32):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.members = []
        img_size = 336
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for cp in checkpoint_paths:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            img_size = int(ck.get("img_size", 336))
            mean = tuple(ck.get("mean", mean)); std = tuple(ck.get("std", std))
            feat_dim = int(ck.get("feat_dim", 768))

            bb = _build_dinov2_backbone(img_size)
            miss, unexp = bb.load_state_dict(ck["backbone"], strict=False)
            if unexp or miss:
                raise RuntimeError(f"DINOv2 backbone mismatch ({Path(cp).name}): missing={miss}, unexpected={unexp}")

            pool = AttentionPool(feat_dim)
            head = nn.Linear(feat_dim, 1)
            head_sd = ck["head"]
            pool.load_state_dict({"score_proj.weight": head_sd["attn_V.weight"], "score_proj.bias": head_sd["attn_V.bias"],
                                  "score_out.weight": head_sd["attn_w.weight"], "score_out.bias": head_sd["attn_w.bias"]})
            head.load_state_dict({"weight": head_sd["head.weight"], "bias": head_sd["head.bias"]})

            self.members.append((bb.to(self.device).eval(), pool.to(self.device).eval(), head.to(self.device).eval()))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        print(f"DINOv2 attention-pooling ensemble: {len(self.members)} member(s) @ {img_size}px")

    @torch.no_grad()
    def predict(self, frames) -> np.ndarray:
        """frames: iterable of HxWx3 uint8 arrays -> (N,) float32 P(neoplasia), soft-voted over members."""
        tensors = [self.transform(Image.fromarray(np.asarray(f)).convert("RGB")) for f in frames]
        member_scores = []
        for backbone, pool, head in self.members:
            outs = []
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    patch_tokens = backbone.forward_features(batch)["x_norm_patchtokens"]
                    pooled = pool(patch_tokens)
                    logit = head(pooled).squeeze(-1)
                outs.append(torch.sigmoid(logit.float()).cpu().numpy())
            member_scores.append(np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32))
        return np.mean(member_scores, axis=0)          # soft voting (mean of per-frame sigmoids)
