
"""Hybrid pose estimation through validated geometric selection.

Combines:
    ImprovedParticleFilter
    GeometricPoseEstimator

This is a selector/fallback hybrid, not an optimal
Bayesian fusion algorithm.
"""

from __future__ import annotations

import time

import numpy as np

from estimators.interfaces import (
    CameraObservation,
    EstimatorConfig,
    EstimatorResult,
)


class HybridPoseEstimator:

    def __init__(
        self,
        particle_filter,
        geometric_estimator,
        *,
        maximum_geometric_error_px: float = 4.0,
        minimum_geometric_inliers: int = 6,
    ):
        if maximum_geometric_error_px <= 0:
            raise ValueError(
                "Maximum geometric error must be positive"
            )

        if minimum_geometric_inliers < 4:
            raise ValueError(
                "Minimum geometric inliers must be >= 4"
            )

        self.particle_filter = particle_filter
        self.geometric_estimator = geometric_estimator

        self.maximum_geometric_error_px = float(
            maximum_geometric_error_px
        )

        self.minimum_geometric_inliers = int(
            minimum_geometric_inliers
        )

        self.initialized = False

    def initialize(
        self,
        config: EstimatorConfig,
        initial_observation: CameraObservation,
    ) -> None:
        self.particle_filter.initialize(
            config,
            initial_observation,
        )

        self.geometric_estimator.initialize(
            config,
            initial_observation,
        )

        self.initialized = True

    def step(
        self,
        observation: CameraObservation,
        dt: float,
    ) -> EstimatorResult:
        if not self.initialized:
            raise RuntimeError(
                "Hybrid estimator not initialized"
            )

        start = time.perf_counter()

        pf_result = self.particle_filter.step(
            observation,
            dt,
        )

        geometric_result = None
        geometric_failure = None

        try:
            geometric_result = self.geometric_estimator.step(
                observation,
                dt,
            )
        except RuntimeError as exc:
            geometric_failure = str(exc)

        use_geometric = False

        if geometric_result is not None:
            diagnostics = geometric_result.diagnostics

            error = diagnostics.get(
                "reprojection_error_px"
            )

            inliers = diagnostics.get(
                "inlier_count", 0
            )

            measurement_success = diagnostics.get(
                "measurement_success", False
            )

            use_geometric = (
                measurement_success
                and error is not None
                and np.isfinite(error)
                and error <= self.maximum_geometric_error_px
                and inliers >= self.minimum_geometric_inliers
            )

        selected = (
            geometric_result
            if use_geometric
            else pf_result
        )

        source = (
            "geometric"
            if use_geometric
            else "improved_pf"
        )

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return EstimatorResult(
            timestamp=observation.timestamp,
            estimator_name="hybrid",
            pose=selected.pose,
            runtime_ms=runtime_ms,
            covariance=selected.covariance,
            diagnostics={
                "selected_estimator": source,
                "geometric_accepted": bool(use_geometric),
                "geometric_failure": geometric_failure,
                "pf_runtime_ms": pf_result.runtime_ms,
                "geometric_runtime_ms": (
                    geometric_result.runtime_ms
                    if geometric_result is not None
                    else None
                ),
                "pf_diagnostics": pf_result.diagnostics,
                "geometric_diagnostics": (
                    geometric_result.diagnostics
                    if geometric_result is not None
                    else None
                ),
            },
        )

    def reset(self):
        self.particle_filter.reset()
        self.geometric_estimator.reset()
        self.initialized = False
