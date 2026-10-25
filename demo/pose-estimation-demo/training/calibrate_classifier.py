
"""Calibrate ANN classifier errors using validation data only.

Usage:
    python -m training.calibrate_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.labels import DEFAULT_DISCRETIZATION
from training.generate_dataset import load_dataset
from training.train_classifier import (
    load_classifier,
    resolve_device,
)


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@torch.inference_mode()
def predict_dataset(
    model,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
):
    """Return predicted labels and uncalibrated confidence."""
    model.eval()

    labels = []
    confidences = []

    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(
            features[start:start + batch_size]
        ).float().to(device)

        logits = model(batch)

        probabilities = torch.softmax(logits, dim=-1)
        confidence, prediction = probabilities.max(dim=1)

        labels.append(prediction.cpu().numpy())
        confidences.append(confidence.cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(confidences),
    )


def calculate_error_samples(
    true_pose_4d: np.ndarray,
    predicted_labels: np.ndarray,
):
    true_pose_4d = np.asarray(
        true_pose_4d, dtype=np.float64
    )

    predicted_labels = np.asarray(
        predicted_labels, dtype=np.int64
    )

    if true_pose_4d.ndim != 2 or true_pose_4d.shape[1] != 4:
        raise ValueError("Expected true poses with shape (N,4)")

    if len(true_pose_4d) != len(predicted_labels):
        raise ValueError("Prediction and truth lengths differ")

    predicted_pose = DEFAULT_DISCRETIZATION.decode(
        predicted_labels
    )

    errors = true_pose_4d - predicted_pose
    errors[:, 3] = wrap_angle(errors[:, 3])

    return errors


def calibration_statistics(errors):
    """Return empirical classifier-error statistics."""
    errors = np.asarray(errors, dtype=np.float64)

    if errors.ndim != 2 or errors.shape[1] != 4:
        raise ValueError("Expected residuals with shape (N,4)")

    if len(errors) < 2 or not np.isfinite(errors).all():
        raise ValueError("At least two finite samples required")

    position_error = np.linalg.norm(errors[:, :3], axis=1)

    return {
        "sample_count": len(errors),
        "mean_error": errors.mean(axis=0).tolist(),
        "std_error": errors.std(axis=0).tolist(),
        "position_error_mean_m": float(position_error.mean()),
        "position_error_p95_m": float(
            np.percentile(position_error, 95)
        ),
        "roll_abs_error_mean_deg": float(
            np.rad2deg(np.abs(errors[:, 3])).mean()
        ),
    }


def sample_error_proposals(
    errors: np.ndarray,
    count: int,
    rng: np.random.Generator,
):
    """Bootstrap empirical pose residuals for particle proposals."""
    errors = np.asarray(errors, dtype=np.float64)

    if errors.ndim != 2 or errors.shape[1] != 4:
        raise ValueError("Expected residual array (N,4)")

    if len(errors) == 0 or count < 0:
        raise ValueError("Invalid proposal count or residual array")

    indices = rng.integers(0, len(errors), size=count)

    return errors[indices].copy()


def calibrate(
    checkpoint_path: str | Path,
    dataset_directory: str | Path,
    output_path: str | Path,
    device: torch.device,
):
    model, checkpoint = load_classifier(
        checkpoint_path, device
    )

    validation = load_dataset(dataset_directory, "val")

    labels, confidences = predict_dataset(
        model, validation["X"], device
    )

    errors = calculate_error_samples(
        validation["pose_4d"], labels
    )

    statistics = calibration_statistics(errors)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(
        output_path,
        errors=errors,
        predicted_labels=labels,
        true_labels=validation["y"],
        confidence=confidences,
        trajectory_id=validation["trajectory_id"],
        checkpoint_epoch=np.array(
            checkpoint["epoch"], dtype=np.int64
        ),
    )

    print("\nValidation error calibration")
    print("-" * 48)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print(f"Saved calibration: {output_path}")

    return statistics


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
        default="checkpoints/classifier_error_calibration.npz",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    calibrate(
        checkpoint_path=args.checkpoint,
        dataset_directory=args.dataset,
        output_path=args.output,
        device=resolve_device(args.device),
    )


if __name__ == "__main__":
    main()
