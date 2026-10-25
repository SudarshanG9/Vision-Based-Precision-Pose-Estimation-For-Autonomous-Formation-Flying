
"""Train the paper-faithful coarse pose classifier.

Usage:
    python -m training.train_classifier --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.pose_classifier import PoseClassifier
from training.generate_dataset import load_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_loader(
    dataset: dict,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    X = torch.from_numpy(dataset["X"]).float()
    y = torch.from_numpy(dataset["y"]).long()

    tensor_dataset = TensorDataset(X, y)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            size = len(targets)
            total_samples += size
            total_loss += float(loss.item()) * size

            predictions = logits.argmax(dim=1)
            total_correct += int(
                (predictions == targets).sum().item()
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_model(
    model: PoseClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    dataset_manifest: dict | None = None,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history = []

    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            size = len(targets)
            train_samples += size
            train_loss += float(loss.item()) * size
            train_correct += int(
                (logits.argmax(dim=1) == targets).sum().item()
            )

        validation = evaluate_epoch(
            model, val_loader, criterion, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_samples,
            "train_accuracy": train_correct / train_samples,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"train_acc={record['train_accuracy']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_acc={record['val_accuracy']:.4f}"
        )

        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": model.input_dim,
                        "hidden_dims": model.hidden_dims,
                        "num_classes": model.num_classes,
                    },
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "validation_accuracy": validation["accuracy"],
                    "seed": seed,
                    "dataset_manifest": dataset_manifest,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def load_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["model_config"]

    model = PoseClassifier(
        input_dim=config["input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        num_classes=config["num_classes"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/training.yaml"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    seed = int(train_cfg["seed"])
    set_seed(seed)

    device = resolve_device(train_cfg["device"])
    print(f"Training device: {device}")

    directory = Path(dataset_cfg["output_directory"])
    train_data = load_dataset(directory, "train")
    val_data = load_dataset(directory, "val")

    with (directory / "manifest.json").open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    model = PoseClassifier(
        input_dim=model_cfg["input_dim"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_classes=model_cfg["output_dim"],
    )

    train_loader = make_loader(
        train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        val_data,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )

    checkpoint_dir = Path(
        train_cfg["checkpoint_directory"]
    )

    checkpoint_path = (
        checkpoint_dir / train_cfg["checkpoint_name"]
    )

    report = train_model(
        model,
        train_loader,
        val_loader,
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["early_stopping_patience"]),
        device=device,
        checkpoint_path=checkpoint_path,
        seed=seed,
        dataset_manifest=manifest,
    )

    with (checkpoint_dir / "training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved best model: {checkpoint_path}")
    print(f"Best epoch: {report['best_epoch']}")


if __name__ == "__main__":
    main()
