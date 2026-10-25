
"""Evaluate the ANN on held-out test trajectories.

Usage:
    python -m training.validate_models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.labels import DEFAULT_DISCRETIZATION
from training.generate_dataset import load_dataset
from training.train_classifier import (
    load_classifier,
    resolve_device,
)
from training.calibrate_classifier import wrap_angle


@torch.inference_mode()
def evaluate_classifier(
    model,
    dataset: dict,
    device: torch.device,
    batch_size: int = 512,
):
    X = dataset["X"]
    y = dataset["y"]
    truth = dataset["pose_4d"]

    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    model.eval()

    predictions = []
    top5_correct = 0
    top10_correct = 0
    total_nll = 0.0

    for start in range(0, len(X), batch_size):
        end = start + batch_size

        features = torch.from_numpy(
            X[start:end]
        ).float().to(device)

        targets = torch.from_numpy(
            y[start:end]
        ).long().to(device)

        logits = model(features)

        loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        )

        total_nll += float(loss.item())

        topk = logits.topk(
            k=min(10, logits.shape[1]), dim=1
        ).indices

        predictions.append(
            topk[:, 0].cpu().numpy()
        )

        top5_correct += int(
            (topk[:, :min(5, topk.shape[1])]
             == targets[:, None]).any(dim=1).sum().item()
        )

        top10_correct += int(
            (topk == targets[:, None]).any(dim=1).sum().item()
        )

    predictions = np.concatenate(predictions)

    decoded = DEFAULT_DISCRETIZATION.decode(predictions)

    position_error = np.linalg.norm(
        decoded[:, :3] - truth[:, :3], axis=1
    )

    roll_error = np.abs(
        wrap_angle(decoded[:, 3] - truth[:, 3])
    )

    bin_distance = DEFAULT_DISCRETIZATION.nearest_class_distance(
        predictions, y
    )

    return {
        "sample_count": int(len(y)),
        "trajectory_count": int(
            len(np.unique(dataset["trajectory_id"]))
        ),
        "exact_accuracy": float(np.mean(predictions == y)),
        "top5_accuracy": top5_correct / len(y),
        "top10_accuracy": top10_correct / len(y),
        "mean_cross_entropy": total_nll / len(y),
        "mean_position_error_m": float(position_error.mean()),
        "median_position_error_m": float(
            np.median(position_error)
        ),
        "p95_position_error_m": float(
            np.percentile(position_error, 95)
        ),
        "mean_roll_error_deg": float(
            np.rad2deg(roll_error).mean()
        ),
        "p95_roll_error_deg": float(
            np.percentile(np.rad2deg(roll_error), 95)
        ),
        "mean_bin_distance": float(bin_distance.mean()),
        "observed_true_classes": int(len(np.unique(y))),
        "observed_predicted_classes": int(
            len(np.unique(predictions))
        ),
        "total_possible_classes": 4800,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pose_classifier_best.pt",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmarks/ann_test_metrics.json",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_classifier(args.checkpoint, device)

    test_data = load_dataset(args.dataset, "test")

    metrics = evaluate_classifier(
        model, test_data, device
    )

    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["evaluation_split"] = "test"

    output = Path(args.output)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)

    print("\nHeld-out ANN evaluation")
    print("-" * 52)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
