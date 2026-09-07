"""Training and prediction loops."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageListDataset, balanced_sampler, eval_transform, train_transform
from .losses import FocalLoss, mixup_loss


def _loader(paths, labels, transform, batch_size, albumentations, workers,
            sampler=None, shuffle=False):
    dataset = ImageListDataset(paths, labels, transform, albumentations)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(shuffle and sampler is None), num_workers=workers, pin_memory=True)


def train_model(model, paths, labels, config, device) -> None:
    """Fit the model in place.

    The schedule is fixed: every run trains for exactly ``config.epochs`` with a cosine-decayed learning
    rate and the final epoch's weights are kept. There is no early stopping and no checkpoint selection.
    """
    model.to(device).train()
    criterion = FocalLoss(gamma=config.gamma, alpha=config.focal_alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda")

    loader = _loader(paths, labels, train_transform(config.img_size), config.batch_size,
                     albumentations=True, workers=config.workers, sampler=balanced_sampler(labels))

    for epoch in range(config.epochs):
        running_loss = 0.0
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                if config.mixup_alpha > 0:
                    lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                    perm = torch.randperm(len(targets), device=device)
                    loss = mixup_loss(criterion, model, images, targets,
                                      images[perm], targets[perm], lam)
                else:
                    loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(targets)
        scheduler.step()
        print(f"  epoch {epoch:02d}  loss={running_loss / len(labels):.4f}", flush=True)


@torch.no_grad()
def predict(model, paths, labels, config, device) -> np.ndarray:
    """Per-frame probabilities. Each frame is scored independently of the others."""
    model.to(device).eval()
    loader = _loader(paths, labels, eval_transform(config.img_size), config.batch_size,
                     albumentations=False, workers=config.workers)
    scores = []
    for images, _, _ in loader:
        with torch.amp.autocast("cuda"):
            logits = model(images.to(device, non_blocking=True))
        scores.append(torch.sigmoid(logits.float()).reshape(-1).cpu().numpy())
    return np.concatenate(scores)
