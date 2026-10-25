
"""Visibility-aware preprocessing for the 42-input ANN."""

from __future__ import annotations

import numpy as np

from estimators.interfaces import (
    CameraObservation,
    NUM_LANDMARKS,
    ANN_INPUT_DIM,
)


def preprocess_keypoints(
    keypoints,
    visibility,
    width: int,
    height: int,
):
    """
    Convert 14 2D landmarks to 42 normalized input values.

    Visible features:
        [u / width, v / height, 1]

    Invisible features:
        [0, 0, 0]
    """
    points = np.asarray(
        keypoints, dtype=np.float64
    )

    mask = np.asarray(
        visibility, dtype=bool
    )

    if points.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            "Expected keypoints with shape (14, 2)"
        )

    if mask.shape != (NUM_LANDMARKS,):
        raise ValueError(
            "Expected visibility with shape (14,)"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image dimensions must be positive"
        )

    if not np.isfinite(points[mask]).all():
        raise ValueError(
            "Visible keypoints must be finite"
        )

    features = np.zeros(
        (NUM_LANDMARKS, 3),
        dtype=np.float32,
    )

    features[mask, 0] = (
        points[mask, 0] / width
    )

    features[mask, 1] = (
        points[mask, 1] / height
    )

    features[:, 2] = mask.astype(np.float32)

    return features.reshape(ANN_INPUT_DIM)


def preprocess_observation(
    observation: CameraObservation,
):
    return preprocess_keypoints(
        keypoints=observation.keypoints,
        visibility=observation.visible,
        width=observation.camera.width,
        height=observation.camera.height,
    )


def preprocess_batch(
    observations: list[CameraObservation],
):
    if not observations:
        return np.empty(
            (0, ANN_INPUT_DIM),
            dtype=np.float32,
        )

    return np.stack(
        [
            preprocess_observation(obs)
            for obs in observations
        ],
        axis=0,
    )


def validate_features(features):
    """Validate the ANN feature matrix."""
    X = np.asarray(features)

    if X.ndim != 2 or X.shape[1] != ANN_INPUT_DIM:
        raise ValueError(
            "Expected feature matrix of shape (N, 42)"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains non-finite values"
        )

    masks = X.reshape(-1, NUM_LANDMARKS, 3)[:, :, 2]

    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError(
            "Visibility features must be binary"
        )

    return True
