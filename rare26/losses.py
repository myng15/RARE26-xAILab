"""Training objective."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., ICCV 2017) on logits.

    ``gamma`` down-weights examples the model already classifies confidently, concentrating the gradient
    on the harder ones. ``alpha`` weights the positive class; the negative class receives ``1 - alpha``.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = (1 - p_t).clamp(min=1e-8) ** self.gamma * bce
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * loss).mean()


def mixup_loss(criterion, model, x_a, y_a, x_b, y_b, lam: float) -> torch.Tensor:
    """MixUp: blend two inputs and combine their losses with the same coefficient."""
    logits = model(lam * x_a + (1.0 - lam) * x_b)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
