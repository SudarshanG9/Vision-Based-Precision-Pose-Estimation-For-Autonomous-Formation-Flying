
"""Improved ANN-guided Particle Filter.

Extends the Phase 4 paper baseline with:

1. Stable log-space measurement likelihood.
2. Gaussian/outlier-mixture reprojection model.
3. ESS-triggered adaptive resampling.
4. Systematic resampling for retained particles.
5. Explicit estimator diagnostics.

This is an experimental extension, not the original paper PF.
"""

from __future__ import annotations

import time

import numpy as np

from estimators.interfaces import EstimatorResult
from estimators.particle_filter_paper import PaperParticleFilter


class ImprovedParticleFilter(PaperParticleFilter):

    def __init__(
        self,
        ann_predictor,
        calibration_errors,
        *,
        measurement_sigma_px: float = 2.0,
        outlier_probability: float = 0.02,
        ess_threshold_ratio: float = 0.5,
        reseed_interval: int = 10,
        particle_sample_limit: int = 250,
    ):
        super().__init__(
            ann_predictor=ann_predictor,
            calibration_errors=calibration_errors,
            reseed_interval=reseed_interval,
            particle_sample_limit=particle_sample_limit,
        )

        if measurement_sigma_px <= 0:
            raise ValueError(
                "measurement_sigma_px must be positive"
            )

        if not 0 <= outlier_probability < 1:
            raise ValueError(
                "outlier_probability must be in [0,1)"
            )

        if not 0 < ess_threshold_ratio <= 1:
            raise ValueError(
                "ess_threshold_ratio must be in (0,1]"
            )

        self.measurement_sigma_px = float(
            measurement_sigma_px
        )
        self.outlier_probability = float(
            outlier_probability
        )
        self.ess_threshold_ratio = float(
            ess_threshold_ratio
        )

        self.resampling_events = 0

    def initialize(self, config, initial_observation):
        super().initialize(config, initial_observation)
        self.resampling_events = 0

    def _measurement_update(self, observation):
        """Stable image measurement likelihood update."""

        visible_mask = observation.visible
        visible_count = int(np.count_nonzero(visible_mask))

        if visible_count == 0:
            return None, 0

        projected, valid = self._project_particles(
            observation
        )

        observed = observation.keypoints[visible_mask]
        predicted = projected[:, visible_mask, :]
        predicted_valid = valid[:, visible_mask]

        residual = predicted - observed[None, :, :]

        # Invalid projections are removed from residual calculations.
        residual = np.where(
            predicted_valid[:, :, None],
            residual,
            0.0,
        )

        squared_error = np.sum(residual**2, axis=-1)

        sigma = self.measurement_sigma_px

        # Gaussian density in the two-dimensional image plane.
        log_normalizer = -np.log(
            2.0 * np.pi * sigma**2
        )

        gaussian_log_density = (
            log_normalizer
            - squared_error / (2.0 * sigma**2)
        )

        # Uniform outlier density over the image plane.
        image_area = (
            observation.camera.width
            * observation.camera.height
        )

        p_outlier = self.outlier_probability

        if p_outlier > 0:
            log_good = (
                np.log1p(-p_outlier)
                + gaussian_log_density
            )

            log_bad = (
                np.log(p_outlier)
                - np.log(image_area)
            )

            feature_log_likelihood = np.logaddexp(
                log_good,
                log_bad,
            )
        else:
            feature_log_likelihood = gaussian_log_density

        # Predicted features outside the camera are unlikely
        # explanations for actually observed features.
        invalid_penalty = -np.log(image_area) - 5.0

        feature_log_likelihood = np.where(
            predicted_valid,
            feature_log_likelihood,
            invalid_penalty,
        )

        log_likelihood = feature_log_likelihood.sum(axis=1)

        # Update existing importance weights in log space.
        log_prior = np.log(
            np.maximum(self.weights, 1e-300)
        )

        log_posterior = log_prior + log_likelihood

        maximum = np.max(log_posterior)

        unnormalized = np.exp(
            log_posterior - maximum
        )

        total = unnormalized.sum()

        if total <= 0 or not np.isfinite(total):
            self.weights.fill(
                1.0 / len(self.weights)
            )
        else:
            self.weights = unnormalized / total

        # Diagnostic RMS reprojection error for the
        # maximum-weight particle, not averaged over all particles.
        best_index = int(np.argmax(self.weights))

        best_errors = squared_error[best_index]
        best_mask = predicted_valid[best_index]

        if np.any(best_mask):
            rms = float(
                np.sqrt(np.mean(best_errors[best_mask]))
            )
        else:
            rms = None

        return rms, visible_count

    def _systematic_indices(self, count: int):
        """Draw weighted particles using systematic resampling."""
        if count == 0:
            return np.empty(0, dtype=np.int64)

        positions = (
            self.rng.random() + np.arange(count)
        ) / count

        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0

        indices = np.searchsorted(
            cumulative,
            positions,
            side="left",
        )

        return np.minimum(
            indices,
            len(self.weights) - 1,
        )

    def _adaptive_resample(
        self,
        observation,
        ann_reseed: bool,
    ):
        """
        Resample from the weighted posterior.

        When ANN reseeding is enabled, a fraction of the particle
        population is replaced with ANN proposals.

        This maintains the project baseline's proposal mechanism.

        Statistical caveat:
        ANN proposal particles are reset to uniform weights here.
        This is a heuristic mixture-resampling method, not an
        exact importance-corrected proposal distribution.
        """
        n = len(self.particles)

        if ann_reseed:
            retained_count = int(
                np.floor(
                    self.config.resampling_alpha * n
                )
            )
        else:
            retained_count = n

        indices = self._systematic_indices(retained_count)

        retained = self.particles[indices].copy()

        new_count = n - retained_count

        if new_count > 0:
            coarse_pose = self._ann_pose(observation)

            proposed = self._sample_ann_particles(
                coarse_pose,
                new_count,
            )

            self.particles = np.concatenate(
                [retained, proposed],
                axis=0,
            )
        else:
            self.particles = retained

        self.weights = np.full(
            n,
            1.0 / n,
            dtype=np.float64,
        )

        self.resampling_events += 1

    def step(self, observation, dt):
        if self.particles is None:
            raise RuntimeError(
                "Improved Particle Filter not initialized"
            )

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        start = time.perf_counter()

        if self.step_index > 0:
            self._propagate(dt)

        reprojection_rms, visible_count = (
            self._measurement_update(observation)
        )

        ess = float(
            1.0 / np.sum(self.weights**2)
        )

        ess_ratio = ess / len(self.particles)

        estimate = self._estimate_pose()

        # Copy the plotted points before any particle replacement.
        sample_count = min(
            self.particle_sample_limit,
            len(self.particles),
        )

        sample_indices = np.linspace(
            0,
            len(self.particles) - 1,
            sample_count,
            dtype=np.int64,
        )

        particle_sample = (
            self.particles[sample_indices, :3]
            .copy()
            .tolist()
        )

        # Reseeding is permitted on designated timesteps.
        ann_reseed = (
            self.step_index % self.reseed_interval == 0
        )

        # On a missing-observation frame, preserve the
        # predicted distribution without forced resampling.
        resample = (
            visible_count > 0
            and ess_ratio < self.ess_threshold_ratio
        )

        if resample:
            self._adaptive_resample(
                observation,
                ann_reseed=ann_reseed,
            )

        self.step_index += 1

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return EstimatorResult(
            timestamp=observation.timestamp,
            estimator_name="improved_pf",
            pose=estimate,
            runtime_ms=runtime_ms,
            diagnostics={
                "effective_sample_size": ess,
                "ess_ratio": ess_ratio,
                "particle_count": len(self.particles),
                "reprojection_rms_px": reprojection_rms,
                "visible_features": visible_count,
                "resampling_performed": bool(resample),
                "ann_reseed_performed": bool(
                    resample and ann_reseed
                ),
                "total_resampling_events": self.resampling_events,
                "particle_sample": particle_sample,
            },
        )

    def reset(self):
        super().reset()
        self.resampling_events = 0
