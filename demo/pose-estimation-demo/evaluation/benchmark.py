
"""Fair comparative benchmarking for continuous pose estimators."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from estimators.interfaces import EstimatorConfig
from estimators.particle_filter_paper import PaperParticleFilter
from estimators.particle_filter_improved import ImprovedParticleFilter
from estimators.geometric_refinement import GeometricPoseEstimator
from estimators.hybrid import HybridPoseEstimator

from evaluation.metrics import (
    evaluate_frame,
    summarize_frames,
)
from scripts.run_experiment import generate_baseline_experiment


ESTIMATOR_NAMES = (
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid",
)


@dataclass
class BenchmarkConfig:
    seed: int = 42
    duration_seconds: float = 2.0
    simulation_fps: int = 20

    particle_count: int = 1000
    resampling_alpha: float = 0.9

    position_noise_std: float = 0.5
    rotation_noise_std: float = 0.01

    pixel_noise_std: float = 1.5

    measurement_sigma_px: float = 2.0
    ess_threshold_ratio: float = 0.5
    outlier_probability: float = 0.02

    paper_reseed_interval: int = 10
    improved_reseed_interval: int = 10

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.duration_seconds <= 0:
            raise ValueError("duration must be positive")
        if self.simulation_fps < 2:
            raise ValueError("simulation_fps must be >= 2")
        if self.particle_count < 1:
            raise ValueError("particle_count must be positive")
        if not 0 <= self.resampling_alpha <= 1:
            raise ValueError("resampling_alpha must be in [0,1]")
        if self.pixel_noise_std < 0:
            raise ValueError("pixel_noise_std must be nonnegative")


def make_estimator(
    name,
    ann_predictor,
    calibration_errors,
    config: BenchmarkConfig,
):
    """Create independent estimator instances for each benchmark."""

    if name == "paper_pf":
        return PaperParticleFilter(
            ann_predictor=ann_predictor,
            calibration_errors=calibration_errors,
            reseed_interval=config.paper_reseed_interval,
            particle_sample_limit=100,
        )

    if name == "improved_pf":
        return ImprovedParticleFilter(
            ann_predictor=ann_predictor,
            calibration_errors=calibration_errors,
            measurement_sigma_px=config.measurement_sigma_px,
            outlier_probability=config.outlier_probability,
            ess_threshold_ratio=config.ess_threshold_ratio,
            reseed_interval=config.improved_reseed_interval,
            particle_sample_limit=100,
        )

    if name == "geometric":
        return GeometricPoseEstimator()

    if name == "hybrid":
        particle_filter = ImprovedParticleFilter(
            ann_predictor=ann_predictor,
            calibration_errors=calibration_errors,
            measurement_sigma_px=config.measurement_sigma_px,
            outlier_probability=config.outlier_probability,
            ess_threshold_ratio=config.ess_threshold_ratio,
            reseed_interval=config.improved_reseed_interval,
            particle_sample_limit=100,
        )

        geometric = GeometricPoseEstimator()

        return HybridPoseEstimator(
            particle_filter,
            geometric,
        )

    raise ValueError(f"Unknown estimator: {name}")


def estimator_configuration(
    name,
    config: BenchmarkConfig,
):
    return EstimatorConfig(
        name=name,
        particle_count=config.particle_count,
        resampling_alpha=config.resampling_alpha,
        position_noise_std=config.position_noise_std,
        rotation_noise_std=config.rotation_noise_std,
        seed=config.seed,
    )


def run_single_estimator(
    name,
    estimator,
    observations,
    truth,
    config: BenchmarkConfig,
):
    """
    Execute one method on fixed observations.

    Initialization is excluded from per-frame runtime.

    Runtime measured within estimator.step() is retained.
    Failed invocations receive separately measured wall time.
    """

    if len(observations) != len(truth):
        raise ValueError("Truth and observations must align")

    estimator.initialize(
        estimator_configuration(name, config),
        observations[0],
    )

    frames = []
    dt = 1.0 / config.simulation_fps

    try:
        for observation, true_pose in zip(
            observations, truth
        ):
            start = time.perf_counter()

            result = None
            failure = None

            try:
                result = estimator.step(observation, dt)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                failure = str(exc)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            frame_metrics = evaluate_frame(
                true_pose,
                result,
                observation,
            )


            # A returned fallback pose is still a numerical estimate,
            # but it is not a newly successful visual measurement.

            measurement_success = None

            if result is not None:
                if name == "geometric":
                    measurement_success = bool(
                        result.diagnostics.get(
                            "measurement_success",
                            False,
                        )
                    )

                elif name == "hybrid":
                    geometric_details = result.diagnostics.get(
                        "geometric_diagnostics"
                    )

                    if isinstance(geometric_details, dict):
                        measurement_success = bool(
                            geometric_details.get(
                                "measurement_success",
                                False,
                            )
                        )


            frame = {

                "measurement_success": measurement_success,

                "frame_index": observation.frame_index,
                "timestamp": observation.timestamp,
                "position_error_m": frame_metrics[
                    "position_error_m"
                ],
                "attitude_error_deg": frame_metrics[
                    "attitude_error_deg"
                ],
                "reprojection_error_px": frame_metrics[
                    "reprojection_error_px"
                ],
                "runtime_ms": (
                    float(result.runtime_ms)
                    if result is not None
                    else elapsed_ms
                ),
                "success": bool(result is not None),
                "failure_reason": failure,
                "estimate": None,
                "diagnostics": {},
            }

            if result is not None:
                frame["estimate"] = {
                    "position": result.pose.position.tolist(),
                    "quaternion": result.pose.quaternion.tolist(),
                    "frame_from": result.pose.frame_from,
                    "frame_to": result.pose.frame_to,
                }

                frame["diagnostics"] = result.diagnostics

            frames.append(frame)
    finally:
        estimator.reset()

    summary = summarize_frames(frames)

    # Include the runtime of failed calls as computational cost.
    attempted_runtimes = [
        frame["runtime_ms"]
        for frame in frames
    ]

    from evaluation.metrics import summarize_values

    summary["attempted_runtime_ms"] = summarize_values(
        attempted_runtimes
    )

    mean_ms = summary["attempted_runtime_ms"]["mean"]

    summary["mean_estimator_fps"] = (
        1000.0 / mean_ms
        if mean_ms is not None and mean_ms > 0
        else None
    )

    return {
        "estimator": name,
        "summary": summary,
        "frames": frames,
    }


def run_benchmark(
    config: BenchmarkConfig,
    ann_predictor,
    calibration_errors,
    estimators=ESTIMATOR_NAMES,
):
    """Benchmark all methods on identical synthetic observations."""

    observations, truth = generate_baseline_experiment(
        seed=config.seed,
        duration=config.duration_seconds,
        fps=config.simulation_fps,
        pixel_noise_std=config.pixel_noise_std,
    )

    results = {}

    for name in estimators:
        estimator = make_estimator(
            name,
            ann_predictor,
            calibration_errors,
            config,
        )

        results[name] = run_single_estimator(
            name,
            estimator,
            observations,
            truth,
            config,
        )

    return {
        "metadata": {
            "seed": config.seed,
            "duration_seconds": config.duration_seconds,
            "simulation_fps": config.simulation_fps,
            "frame_count": len(observations),
            "particle_count": config.particle_count,
            "resampling_alpha": config.resampling_alpha,
            "pixel_noise_std": config.pixel_noise_std,
            "scenario": "distribution_compatible_approach",
            "scenario_origin": "Phase 4 baseline generator",
            "evaluation_type": "synthetic_research_benchmark",
        },
        "ground_truth": [
            {
                "frame_index": index,
                "timestamp": obs.timestamp,
                "position": pose.position.tolist(),
                "quaternion": pose.quaternion.tolist(),
            }
            for index, (obs, pose) in enumerate(
                zip(observations, truth)
            )
        ],
        "observations": [
            {
                "frame_index": obs.frame_index,
                "timestamp": obs.timestamp,
                "keypoints": [
                    pixel.tolist() if visible else None
                    for pixel, visible in zip(
                        obs.keypoints, obs.visible
                    )
                ],
                "visible": obs.visible.tolist(),
            }
            for obs in observations
        ],
        "results": results,
    }
