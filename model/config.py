"""Hyperparameters of the submitted configuration.

These values were selected by a search over learning rate, weight decay, focal gamma, augmentation
strength, MixUp coefficient, LoRA rank and adapter placement, and re-validated afterwards on a
held-out slice of the pooled training data.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    img_size: int = 336
    batch_size: int = 32
    epochs: int = 20
    workers: int = 6

    lr: float = 3.6249353779145154e-4
    weight_decay: float = 3.9628545145815636e-05

    gamma: float = 2.6602225177473289
    focal_alpha: float = 0.25
    mixup_alpha: float = 0.41

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    pooling: str = "attention"
    seeds: tuple = (42, 43, 44, 45, 46)


DEFAULT = Config()
