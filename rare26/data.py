"""Datasets, transforms and split loading."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(split_csv: Path, data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (absolute image paths, labels) for one ``split`` value of a split CSV."""
    df = pd.read_csv(split_csv)
    df = df[df.split == split]
    if df.empty:
        raise ValueError(f"split '{split}' is empty in {split_csv}")
    paths = np.array([str(Path(data_root) / p) for p in df.image_path])
    return paths, df.target.to_numpy(dtype=int)


def train_transform(img_size: int):
    """Geometric augmentation plus mild photometric jitter.

    Early neoplasia presents as a subtle colour and texture change, so the photometric range is kept
    narrow; aggressive colour or blur augmentation risks destroying the signal being learned.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.06, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(img_size: int):
    """Deterministic resize and normalise, used for validation and for inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform, albumentations: bool):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.transform = transform
        self.albumentations = albumentations

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        if self.albumentations:
            image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=image)["image"]
        else:
            tensor = self.transform(Image.open(path).convert("RGB"))
        return tensor, int(self.labels[idx]), path


def balanced_sampler(labels) -> WeightedRandomSampler:
    """Sample the two classes with equal probability.

    Positives are roughly 5% of the training set, so without this the minority class contributes very
    few gradients per epoch.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)
