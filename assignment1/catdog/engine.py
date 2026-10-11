"""Training and evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int
    labels: list[int]
    predictions: list[int]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    amp_enabled = device.type == "cuda"

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    return EpochMetrics(
        loss=running_loss / total,
        accuracy=correct / total,
        correct=correct,
        total=total,
        labels=all_labels,
        predictions=all_predictions,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> EpochMetrics:
    return _run_epoch(model, loader, criterion, device, optimizer, scaler)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    return _run_epoch(model, loader, criterion, device)
