
"""Verify the pose estimation research environment.

Run from pose-estimation-demo/:

    python -m scripts.verify_installation

Optional strict verification:

    python -m scripts.verify_installation --strict

Strict mode requires trained model and calibration artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np


MODULES = (
    "numpy",
    "torch",
    "cv2",
    "scipy",
    "yaml",
    "fastapi",
    "pydantic",
    "uvicorn",
    "pytest",
)

PROJECT_MODULES = (
    "estimators.interfaces",
    "estimators.particle_filter_paper",
    "estimators.particle_filter_improved",
    "estimators.geometric_refinement",
    "estimators.hybrid",
    "simulation.transforms",
    "simulation.aircraft_model",
    "simulation.camera",
    "simulation.trajectory",
    "simulation.renderer",
    "simulation.observations",
    "simulation.scenarios",
    "models.labels",
    "models.preprocessing",
    "models.pose_classifier",
    "training.generate_dataset",
    "training.train_classifier",
    "training.calibrate_classifier",
    "training.validate_models",
    "evaluation.metrics",
    "evaluation.benchmark",
    "evaluation.ablation",
    "evaluation.report",
    "services.model_registry",
    "services.experiment_runner",
    "services.experiment_store",
    "services.telemetry_builder",
    "api.main",
)

MODEL_PATH = Path(
    "checkpoints/pose_classifier_best.pt"
)

CALIBRATION_PATH = Path(
    "checkpoints/classifier_error_calibration.npz"
)

DATASET_DIRECTORY = Path("data/processed")


def check_imports(module_names):
    results = {}

    for name in module_names:
        try:
            importlib.import_module(name)
            results[name] = {
                "status": "passed",
                "error": None,
            }
        except Exception as exc:
            results[name] = {
                "status": "failed",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    return results


def check_aircraft():
    from simulation.aircraft_model import (
        create_aircraft_model,
    )

    aircraft = create_aircraft_model()

    assert aircraft.landmarks.shape == (14, 3)
    assert len(aircraft.names) == 14
    assert len(set(aircraft.names)) == 14

    assert np.isfinite(aircraft.landmarks).all()

    return {
        "landmark_count": 14,
        "geometry_extent_m": (
            aircraft.bounding_extent().tolist()
        ),
    }


def check_camera():
    from estimators.interfaces import Pose3D
    from simulation.camera import default_camera

    camera = default_camera()

    pose = Pose3D(
        position=[40.0, 0.0, 0.0],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        frame_from="L",
        frame_to="F",
    )

    pixels, visible, _ = camera.project_landmarks(
        np.array([[0.0, 0.0, 0.0]]),
        pose,
    )

    assert visible[0]

    np.testing.assert_allclose(
        pixels[0],
        [320.0, 240.0],
        atol=1e-8,
    )

    return {
        "projection_valid": True,
        "optical_center": pixels[0].tolist(),
    }


def check_dataset():
    from training.generate_dataset import load_dataset

    results = {}

    for split in ("train", "val", "test"):
        path = DATASET_DIRECTORY / f"{split}.npz"

        if not path.is_file():
            results[split] = {
                "status": "missing",
            }
            continue

        data = load_dataset(
            DATASET_DIRECTORY,
            split,
        )

        assert data["X"].ndim == 2
        assert data["X"].shape[1] == 42
        assert len(data["X"]) == len(data["y"])
        assert np.isfinite(data["X"]).all()

        results[split] = {
            "status": "passed",
            "samples": int(len(data["y"])),
            "unique_classes": int(
                len(np.unique(data["y"]))
            ),
            "trajectory_ids": np.unique(
                data["trajectory_id"]
            ).tolist(),
        }

    # Ensure complete trajectories do not cross partitions.
    available = [
        name for name in results
        if results[name]["status"] == "passed"
    ]

    for i, first in enumerate(available):
        for second in available[i + 1:]:
            a = set(results[first]["trajectory_ids"])
            b = set(results[second]["trajectory_ids"])

            assert a.isdisjoint(b), (
                f"Trajectory leakage: {first}/{second}"
            )

    return results


def check_checkpoint():
    if not MODEL_PATH.is_file():
        return {"status": "missing"}

    import torch

    from models.pose_classifier import PoseClassifier

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["model_config"]

    model = PoseClassifier(
        input_dim=int(config["input_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        num_classes=int(config["num_classes"]),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )
    model.eval()

    with torch.inference_mode():
        logits = model(torch.zeros(1, 42))

    assert logits.shape == (1, 4800)
    assert torch.isfinite(logits).all()

    return {
        "status": "passed",
        "epoch": int(checkpoint["epoch"]),
        "classes": 4800,
    }


def check_calibration():
    if not CALIBRATION_PATH.is_file():
        return {"status": "missing"}

    with np.load(
        CALIBRATION_PATH,
        allow_pickle=False,
    ) as archive:
        errors = archive["errors"]

        assert errors.ndim == 2
        assert errors.shape[1] == 4
        assert len(errors) > 0
        assert np.isfinite(errors).all()

        return {
            "status": "passed",
            "samples": int(len(errors)),
            "dimensions": 4,
        }


def run_verification(strict=False):
    report = {
        "python": sys.version.split()[0],
        "dependencies": check_imports(MODULES),
        "project_modules": check_imports(
            PROJECT_MODULES
        ),
        "checks": {},
    }

    checks = {
        "aircraft": check_aircraft,
        "camera": check_camera,
        "dataset": check_dataset,
        "checkpoint": check_checkpoint,
        "calibration": check_calibration,
    }

    for name, function in checks.items():
        try:
            report["checks"][name] = {
                "status": "passed",
                "details": function(),
            }
        except Exception as exc:
            report["checks"][name] = {
                "status": "failed",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    failures = []

    for group_name in ("dependencies", "project_modules"):
        for name, result in report[group_name].items():
            if result["status"] != "passed":
                failures.append(f"{group_name}: {name}")

    for name, result in report["checks"].items():
        if result["status"] == "failed":
            failures.append(f"check: {name}")

    if strict:
        for name in ("checkpoint", "calibration"):
            details = report["checks"].get(
                name, {}
            ).get("details", {})

            if details.get("status") != "passed":
                failures.append(
                    f"required artifact: {name}"
                )

        dataset = report["checks"].get(
            "dataset", {}
        ).get("details", {})

        for split in ("train", "val", "test"):
            if dataset.get(
                split, {}
            ).get("status") != "passed":
                failures.append(
                    f"required dataset: {split}"
                )

    report["success"] = len(failures) == 0
    report["failures"] = failures

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

    report = run_verification(
        strict=args.strict
    )

    print(
        json.dumps(report, indent=2)
    )

    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
