"""Train the single-view Real-IAD classification and segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import RealIADSingleViewModel
from realiad_dataset import make_dataloader


def choose_device() -> torch.device:
    """Use CUDA, then Apple Silicon MPS, and otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Measure mask overlap; zero is a perfect match."""
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    total = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = (2.0 * intersection + 1.0) / (total + 1.0)
    return 1.0 - dice.mean()


def calculate_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    mask_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine image classification loss with pixel-level mask loss."""
    classification_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["classification_logits"], labels
    )
    mask_bce = nn.functional.binary_cross_entropy_with_logits(
        outputs["mask_logits"], masks
    )
    segmentation_loss = mask_bce + dice_loss(outputs["mask_logits"], masks)
    total_loss = classification_loss + mask_weight * segmentation_loss
    return total_loss, classification_loss, segmentation_loss


def set_training_stage(model: RealIADSingleViewModel, fine_tune: bool) -> None:
    """Freeze the encoder, or unfreeze only its final two stages."""
    encoder_parts = [
        model.stem,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]
    for part in encoder_parts:
        for parameter in part.parameters():
            parameter.requires_grad = False

    if fine_tune:
        for part in (model.layer3, model.layer4):
            for parameter in part.parameters():
                parameter.requires_grad = True


def make_optimizer(
    model: RealIADSingleViewModel, learning_rate: float, fine_tune: bool
) -> torch.optim.Optimizer:
    """Give pretrained encoder parameters a smaller learning rate."""
    encoder_parameters = list(model.layer3.parameters()) + list(model.layer4.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups: list[dict] = [{"params": new_parameters, "lr": learning_rate}]
    if fine_tune:
        groups.append({"params": encoder_parameters, "lr": learning_rate * 0.1})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def run_epoch(
    model: RealIADSingleViewModel,
    loader,
    device: torch.device,
    mask_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and return averaged metrics."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "classification_loss": 0.0, "segmentation_loss": 0.0}
    correct = examples = batches = 0
    true_positive = false_positive = false_negative = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss, classification_loss, segmentation_loss = calculate_loss(
                outputs, labels, masks, mask_weight
            )
            if training:
                loss.backward()
                optimizer.step()

        totals["loss"] += loss.item()
        totals["classification_loss"] += classification_loss.item()
        totals["segmentation_loss"] += segmentation_loss.item()
        batches += 1

        predicted_labels = outputs["classification_logits"] >= 0
        correct += (predicted_labels == labels.bool()).sum().item()
        examples += labels.numel()

        predicted_masks = outputs["mask_logits"] >= 0
        target_masks = masks.bool()
        true_positive += (predicted_masks & target_masks).sum().item()
        false_positive += (predicted_masks & ~target_masks).sum().item()
        false_negative += (~predicted_masks & target_masks).sum().item()

    if batches == 0:
        raise RuntimeError("The data loader did not produce any batches.")

    metrics = {name: value / batches for name, value in totals.items()}
    metrics["accuracy"] = correct / examples
    metrics["dice"] = (2 * true_positive + 1) / (
        2 * true_positive + false_positive + false_negative + 1
    )
    return metrics


def print_metrics(epoch: int, split: str, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch:02d} split={split:<5} "
        f"loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f} "
        f"dice={metrics['dice']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frozen-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best_single_view.pt")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()
    print(f"device={device}")

    train_loader = make_dataloader("train", args.batch_size, args.num_workers)
    val_loader = make_dataloader("val", args.batch_size, args.num_workers)
    model = RealIADSingleViewModel(pretrained=not args.no_pretrained).to(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    fine_tune = False
    set_training_stage(model, fine_tune=False)
    optimizer = make_optimizer(model, args.learning_rate, fine_tune=False)

    for epoch in range(1, args.epochs + 1):
        should_fine_tune = epoch > args.frozen_epochs
        if should_fine_tune != fine_tune:
            fine_tune = should_fine_tune
            set_training_stage(model, fine_tune=True)
            optimizer = make_optimizer(model, args.learning_rate, fine_tune=True)
            print("unfroze ResNet layer3 and layer4")

        train_metrics = run_epoch(
            model, train_loader, device, args.mask_weight, optimizer, args.max_batches
        )
        validation_metrics = run_epoch(
            model, val_loader, device, args.mask_weight, max_batches=args.max_batches
        )
        print_metrics(epoch, "train", train_metrics)
        print_metrics(epoch, "val", validation_metrics)

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
