
"""Build synchronized, JSON-safe dashboard telemetry."""

from __future__ import annotations

import math

import numpy as np


def json_safe(value):
    """Recursively convert NumPy and non-finite values to JSON types."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return json_safe(float(value))

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (str, int, bool)):
        return value

    raise TypeError(
        f"Unsupported telemetry value: {type(value)!r}"
    )


def build_telemetry(
    experiment_id,
    benchmark,
    ann_results=None,
    playback_fps=30,
):
    """Convert one benchmark into frame-aligned dashboard telemetry."""

    metadata = benchmark["metadata"]
    ground_truth = benchmark["ground_truth"]
    observations = benchmark["observations"]
    results = benchmark["results"]

    count = metadata["frame_count"]

    if len(ground_truth) != count:
        raise ValueError("Ground-truth frame count mismatch")

    if len(observations) != count:
        raise ValueError("Observation frame count mismatch")

    if ann_results is not None and len(ann_results) != count:
        raise ValueError("ANN frame count mismatch")

    result_maps = {}

    for name, experiment in results.items():
        frames = experiment["frames"]

        if len(frames) != count:
            raise ValueError(
                f"Frame count mismatch for estimator {name}"
            )

        mapping = {
            frame["frame_index"]: frame
            for frame in frames
        }

        if len(mapping) != count:
            raise ValueError(
                f"Duplicate estimator frames: {name}"
            )

        result_maps[name] = mapping

    frames = []

    for index in range(count):
        observation = observations[index]
        truth = ground_truth[index]

        if (
            observation["frame_index"] != index
            or truth["frame_index"] != index
        ):
            raise ValueError(
                "Unaligned observation and ground-truth frame"
            )

        estimates = {}
        errors = {}
        diagnostics = {}

        for name, mapping in result_maps.items():
            result = mapping[index]

            if not np.isclose(
                result["timestamp"],
                observation["timestamp"],
                atol=1e-8,
            ):
                raise ValueError(
                    f"Timestamp mismatch: {name}, frame {index}"
                )

            estimates[name] = (
                result["estimate"]
                if result["success"]
                else None
            )

            errors[name] = {
                "position_error_m": result[
                    "position_error_m"
                ],
                "attitude_error_deg": result[
                    "attitude_error_deg"
                ],
                "reprojection_error_px": result[
                    "reprojection_error_px"
                ],
            }

            diagnostics[name] = {
                **result["diagnostics"],
                "runtime_ms": result["runtime_ms"],
                "success": result["success"],
                "failure_reason": result["failure_reason"],
            }

        frame = {
            "frame_index": index,
            "timestamp": observation["timestamp"],
            "observation": {
                "keypoints": observation["keypoints"],
                "visible": observation["visible"],
            },
            "ground_truth": {
                "position": truth["position"],
                "quaternion": truth["quaternion"],
            },
            "ann": (
                ann_results[index]
                if ann_results is not None
                else None
            ),
            "estimates": estimates,
            "errors": errors,
            "diagnostics": diagnostics,
        }

        frames.append(json_safe(frame))

    return json_safe({
        "experiment_id": experiment_id,
        "scenario_id": metadata["scenario"],
        "simulation_fps": metadata["simulation_fps"],
        "playback_fps": playback_fps,
        "frame_count": count,
        "frames": frames,
    })


def build_experiment_summary(benchmark, model_metadata):
    """Create the summary served to the frontend."""

    summaries = {
        name: result["summary"]
        for name, result in benchmark["results"].items()
    }

    return json_safe({
        "metrics": summaries,
        "runtime": {
            name: summary["attempted_runtime_ms"]
            for name, summary in summaries.items()
        },
        "metadata": {
            **benchmark["metadata"],
            "model": model_metadata,
        },
    })
