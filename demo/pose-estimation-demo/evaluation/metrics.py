
"""Common scientific evaluation metrics.

Translations: meters
Rotations: radians internally, degrees in reports
Reprojection: pixels
Runtime: milliseconds
"""

from __future__ import annotations

import numpy as np

from estimators.interfaces import Pose3D
from simulation.transforms import rotation_error_degrees


def position_error(
    truth: Pose3D,
    estimate: Pose3D,
) -> float:
    _check_frames(truth, estimate)
    return float(
        np.linalg.norm(
            truth.position - estimate.position
        )
    )


def attitude_error(
    truth: Pose3D,
    estimate: Pose3D,
) -> float:
    _check_frames(truth, estimate)
    return rotation_error_degrees(truth, estimate)


def _check_frames(truth, estimate):
    if (
        truth.frame_from != estimate.frame_from
        or truth.frame_to != estimate.frame_to
    ):
        raise ValueError(
            "Truth and estimate must use identical frames"
        )


def summarize_values(values):
    """Summarize finite measurements; invalid values are excluded."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]

    if len(finite) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("Expected a nonempty weight vector")

    if not np.isfinite(weights).all():
        raise ValueError("Weights must be finite")

    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid particle weights")

    normalized = weights / weights.sum()

    return float(1.0 / np.sum(normalized**2))


def evaluate_frame(truth, result, observation=None):
    """Evaluate one estimator result against ground truth."""
    if result is None:
        return {
            "position_error_m": None,
            "attitude_error_deg": None,
            "runtime_ms": None,
            "reprojection_error_px": None,
            "success": False,
        }

    pos = position_error(truth, result.pose)
    att = attitude_error(truth, result.pose)

    reprojection = None

    if observation is not None:
        from estimators.geometric_refinement import (
            pose_reprojection_error,
        )

        reprojection = pose_reprojection_error(
            observation,
            result.pose,
        )

    return {
        "position_error_m": pos,
        "attitude_error_deg": att,
        "runtime_ms": float(result.runtime_ms),
        "reprojection_error_px": reprojection,
        "success": True,
    }


def summarize_frames(frames):
    """Compute accuracy, failure, and runtime summaries."""
    count = len(frames)

    if count == 0:
        raise ValueError("Cannot summarize zero frames")

    successful = sum(
        bool(frame["success"])
        for frame in frames
    )

    result = {
        "frame_count": count,
        "successful_frames": successful,
        "failed_frames": count - successful,
        "failure_rate": (count - successful) / count,
    }

    fields = (
        "position_error_m",
        "attitude_error_deg",
        "runtime_ms",
        "reprojection_error_px",
    )

    for field in fields:
        values = [
            np.nan if frame.get(field) is None
            else frame[field]
            for frame in frames
        ]

        result[field] = summarize_values(values)

    return result


def paired_position_difference(
    first_frames,
    second_frames,
):
    """
    Compare two methods on matching frame indices.

    Positive difference means the first method has larger error.

    Only frames with valid results from both methods are included.
    """
    first = {
        frame["frame_index"]: frame
        for frame in first_frames
    }

    second = {
        frame["frame_index"]: frame
        for frame in second_frames
    }

    common = sorted(set(first) & set(second))

    differences = []

    for index in common:
        a = first[index].get("position_error_m")
        b = second[index].get("position_error_m")

        if a is None or b is None:
            continue

        differences.append(float(a - b))

    return summarize_values(differences)
