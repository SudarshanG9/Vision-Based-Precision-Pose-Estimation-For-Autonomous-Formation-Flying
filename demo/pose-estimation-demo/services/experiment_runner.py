
"""Coordinate experiment execution, evaluation, and persistence."""

from __future__ import annotations


from dataclasses import dataclass

import numpy as np

import hashlib
from pathlib import Path


from evaluation.benchmark import (
    BenchmarkConfig,
    ESTIMATOR_NAMES,
    run_benchmark,
)
from evaluation.metrics import summarize_values
from models.labels import decode_pose
from services.telemetry_builder import (
    build_telemetry,
    build_experiment_summary,
)


SUPPORTED_BENCHMARK_SCENARIOS = (
    "distribution_compatible_approach",
)


@dataclass(frozen=True)
class ExperimentRequestConfig:
    scenario_id: str
    estimators: tuple[str, ...]

    particle_count: int = 1000
    resampling_alpha: float = 0.9
    pixel_noise_std: float = 1.5
    seed: int = 42

    duration_seconds: float = 2.0
    simulation_fps: int = 20

    position_noise_std: float = 0.5
    rotation_noise_std: float = 0.01

    measurement_sigma_px: float = 2.0
    ess_threshold_ratio: float = 0.5

    def __post_init__(self):
        if self.scenario_id not in SUPPORTED_BENCHMARK_SCENARIOS:
            raise ValueError(
                "Unsupported benchmark scenario"
            )

        if not self.estimators:
            raise ValueError(
                "At least one estimator is required"
            )

        if len(set(self.estimators)) != len(self.estimators):
            raise ValueError(
                "Duplicate estimator names"
            )

        unknown = set(self.estimators) - set(ESTIMATOR_NAMES)

        if unknown:
            raise ValueError(
                f"Unsupported estimators: {sorted(unknown)}"
            )

    def benchmark_config(self):
        return BenchmarkConfig(
            seed=self.seed,
            duration_seconds=self.duration_seconds,
            simulation_fps=self.simulation_fps,
            particle_count=self.particle_count,
            resampling_alpha=self.resampling_alpha,
            pixel_noise_std=self.pixel_noise_std,
            position_noise_std=self.position_noise_std,
            rotation_noise_std=self.rotation_noise_std,
            measurement_sigma_px=self.measurement_sigma_px,
            ess_threshold_ratio=self.ess_threshold_ratio,
        )


def file_sha256(path):
    """Calculate a reproducible fingerprint of a model artifact."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


class ExperimentRunner:
    def __init__(self, model_registry, experiment_store):
        self.models = model_registry
        self.store = experiment_store

    def execute(self, experiment_id: str):
        """
        Execute a pending experiment.

        Called by FastAPI BackgroundTasks in the local backend.
        Errors are reflected in experiment status.
        """
        record = self.store.get(experiment_id)

        if record["status"] != "pending":
            raise RuntimeError(
                "Only pending experiments can execute"
            )

        self.store.update_status(
            experiment_id,
            "running",
            progress=0.05,
        )

        try:
            request = ExperimentRequestConfig(
                **{
                    **record["configuration"],
                    "estimators": tuple(
                        record["configuration"]["estimators"]
                    ),
                }
            )

            benchmark_config = request.benchmark_config()

            self.models.load()

            predictor = self.models.predict
            calibration = self.models.calibration_errors()

            self.store.update_status(
                experiment_id,
                "running",
                progress=0.20,
            )

            benchmark = run_benchmark(
                config=benchmark_config,
                ann_predictor=predictor,
                calibration_errors=calibration,
                estimators=request.estimators,
            )

            self.store.update_status(
                experiment_id,
                "running",
                progress=0.80,
            )

            ann_results = self._ann_frame_predictions(
                benchmark
            )

            telemetry = build_telemetry(
                experiment_id=experiment_id,
                benchmark=benchmark,
                ann_results=ann_results,
                playback_fps=30,
            )

            
            model_metadata = self.models.metadata()

            checkpoint_path = getattr(
                self.models,
                "checkpoint_path",
                None,
            )

            calibration_path = getattr(
                self.models,
                "calibration_path",
                None,
            )

            if checkpoint_path is not None:
                model_metadata["checkpoint_sha256"] = file_sha256(
                    checkpoint_path
                )

            if calibration_path is not None:
                model_metadata["calibration_sha256"] = file_sha256(
                    calibration_path
                )

            summary = build_experiment_summary(
                benchmark=benchmark,
                model_metadata=model_metadata,
            )


            self.store.complete(
                experiment_id,
                summary=summary,
                telemetry=telemetry,
            )

        except Exception as exc:
            self.store.update_status(
                experiment_id,
                "failed",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise

    def _ann_frame_predictions(self, benchmark):
        """
        Recover per-frame ANN predictions for display.

        Benchmark observations are already normalized 2D
        measurements, but the registry requires CameraObservation.
        """

        from estimators.interfaces import (
            CameraIntrinsics,
            CameraObservation,
        )
        from simulation.camera import forward_camera_mount

        camera = CameraIntrinsics(
            fx=800,
            fy=800,
            cx=320,
            cy=240,
            width=640,
            height=480,
        )

        mount = forward_camera_mount()

        results = []

        for record in benchmark["observations"]:
            visible = np.asarray(
                record["visible"],
                dtype=bool,
            )

            keypoints = np.full(
                (14, 2),
                np.nan,
                dtype=np.float64,
            )

            for index, pixel in enumerate(
                record["keypoints"]
            ):
                if visible[index]:
                    keypoints[index] = pixel

            observation = CameraObservation(
                timestamp=record["timestamp"],
                frame_index=record["frame_index"],
                keypoints=keypoints,
                visible=visible,
                camera=camera,
                camera_pose=mount,
            )

            results.append(
                self.models.predict_details(observation)
            )

        return results
