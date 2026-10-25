
"""Lazy model loading and shared ANN inference services."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

import numpy as np
import torch

from models.labels import decode_pose
from models.preprocessing import preprocess_observation
from training.train_classifier import (
    load_classifier,
    resolve_device,
)


class ModelRegistry:
    def __init__(
        self,
        checkpoint_path="checkpoints/pose_classifier_best.pt",
        calibration_path="checkpoints/classifier_error_calibration.npz",
        device="auto",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.calibration_path = Path(calibration_path)
        self.device_name = device

        self._lock = RLock()

        self._model = None
        self._device = None
        self._checkpoint = None
        self._calibration = None

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        """Load models only when first required."""
        with self._lock:
            if self.loaded:
                return

            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing ANN checkpoint: "
                    f"{self.checkpoint_path}"
                )

            if not self.calibration_path.is_file():
                raise FileNotFoundError(
                    f"Missing calibration artifact: "
                    f"{self.calibration_path}"
                )

            device = resolve_device(self.device_name)

            model, checkpoint = load_classifier(
                self.checkpoint_path,
                device,
            )

            with np.load(
                self.calibration_path,
                allow_pickle=False,
            ) as archive:
                errors = archive["errors"].copy()

            if (
                errors.ndim != 2
                or errors.shape[1] != 4
                or len(errors) == 0
                or not np.isfinite(errors).all()
            ):
                raise ValueError(
                    "Invalid classifier calibration artifact"
                )

            self._model = model
            self._device = device
            self._checkpoint = checkpoint
            self._calibration = errors

    def predict(self, observation):
        """Predict a coarse class from one camera observation."""
        self.load()

        features = preprocess_observation(observation)

        with self._lock, torch.inference_mode():
            tensor = torch.from_numpy(
                features[None, :]
            ).to(self._device)

            logits = self._model(tensor)
            label = int(logits.argmax(dim=1).item())

        return label

    def predict_details(self, observation):
        """Return class ID, confidence, and decoded pose."""
        self.load()

        features = preprocess_observation(observation)

        with self._lock, torch.inference_mode():
            tensor = torch.from_numpy(
                features[None, :]
            ).to(self._device)

            logits = self._model(tensor)

            probabilities = torch.softmax(
                logits, dim=1
            )

            confidence, label = probabilities.max(dim=1)

            class_id = int(label.item())
            class_confidence = float(confidence.item())

        pose_4d = decode_pose(class_id)

        return {
            "class_id": class_id,
            "confidence": class_confidence,
            "coarse_position": pose_4d[:3].tolist(),
            "coarse_roll_rad": float(pose_4d[3]),
        }

    def calibration_errors(self):
        self.load()
        return self._calibration.copy()

    def metadata(self):
        if not self.loaded:
            return {
                "loaded": False,
                "checkpoint": str(self.checkpoint_path),
            }

        return {
            "loaded": True,
            "checkpoint": str(self.checkpoint_path),
            "calibration": str(self.calibration_path),
            "checkpoint_epoch": int(
                self._checkpoint["epoch"]
            ),
            "device": str(self._device),
        }
