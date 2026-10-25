
"""Consolidated Phase 5 tests.

Run:
    python -m pytest tests/test_geometric_refinement.py -v
"""

import numpy as np
import pytest

from estimators.interfaces import (
    CameraObservation,
    EstimatorConfig,
    Pose3D,
)
from estimators.particle_filter_improved import (
    ImprovedParticleFilter,
)
from estimators.geometric_refinement import (
    GeometricPoseEstimator,
    pose_reprojection_error,
)
from estimators.hybrid import HybridPoseEstimator

from models.labels import encode_pose
from simulation.aircraft_model import create_aircraft_model
from simulation.camera import default_camera
from simulation.renderer import render_aircraft
from simulation.observations import (
    ObservationConfig,
    generate_observation,
)
from simulation.transforms import quaternion_to_euler


@pytest.fixture
def synthetic_observation():
    aircraft = create_aircraft_model()
    camera = default_camera()

    truth = Pose3D(
        position=[45.0, 2.0, -1.0],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        frame_from="L",
        frame_to="F",
    )

    rendered = render_aircraft(
        aircraft,
        camera,
        truth,
    )

    observation = generate_observation(
        render=rendered,
        camera=camera,
        timestamp=0.0,
        frame_index=0,
        rng=np.random.default_rng(42),
        config=ObservationConfig(
            pixel_noise_std=0.0,
            dropout_probability=0.0,
        ),
    )

    return observation, truth


def make_predictor(truth):
    """TEST-ONLY oracle coarse-class predictor."""

    def predict(observation):
        roll = quaternion_to_euler(
            truth.quaternion
        )[0]

        return encode_pose([
            truth.position[0],
            truth.position[1],
            truth.position[2],
            roll,
        ])

    return predict


def make_improved_filter(truth, seed=42):
    rng = np.random.default_rng(123)

    residuals = np.column_stack([
        rng.normal(0, 1.0, 100),
        rng.normal(0, 1.0, 100),
        rng.normal(0, 0.5, 100),
        rng.normal(0, 0.02, 100),
    ])

    estimator = ImprovedParticleFilter(
        ann_predictor=make_predictor(truth),
        calibration_errors=residuals,
        measurement_sigma_px=2.0,
        outlier_probability=0.02,
        ess_threshold_ratio=0.8,
        reseed_interval=2,
        particle_sample_limit=10,
    )

    config = EstimatorConfig(
        name="improved_pf",
        particle_count=128,
        resampling_alpha=0.9,
        position_noise_std=0.1,
        rotation_noise_std=0.01,
        seed=seed,
    )

    return estimator, config


def test_improved_filter_initialization(synthetic_observation):
    observation, truth = synthetic_observation

    estimator, config = make_improved_filter(truth)

    estimator.initialize(config, observation)

    assert estimator.particles.shape == (128, 7)
    assert estimator.weights.shape == (128,)

    np.testing.assert_allclose(
        estimator.weights.sum(),
        1.0,
    )


def test_improved_measurement_weights(synthetic_observation):
    observation, truth = synthetic_observation

    estimator, config = make_improved_filter(truth)
    estimator.initialize(config, observation)

    rms, visible_count = estimator._measurement_update(
        observation
    )

    assert visible_count > 0
    assert rms is None or rms >= 0

    assert np.isfinite(estimator.weights).all()

    np.testing.assert_allclose(
        estimator.weights.sum(),
        1.0,
        atol=1e-10,
    )


def test_adaptive_resampling(synthetic_observation):
    observation, truth = synthetic_observation

    estimator, config = make_improved_filter(truth)
    estimator.initialize(config, observation)

    estimator._measurement_update(observation)

    estimator._adaptive_resample(
        observation,
        ann_reseed=True,
    )

    assert estimator.particles.shape == (128, 7)
    assert estimator.resampling_events == 1

    np.testing.assert_allclose(
        estimator.weights,
        np.full(128, 1 / 128),
    )


def test_improved_filter_step(synthetic_observation):
    observation, truth = synthetic_observation

    estimator, config = make_improved_filter(truth)
    estimator.initialize(config, observation)

    result = estimator.step(observation, 0.02)

    assert result.estimator_name == "improved_pf"

    assert np.isfinite(result.pose.position).all()
    assert np.isfinite(result.pose.quaternion).all()

    assert 0 < result.diagnostics["ess_ratio"] <= 1.000001


def test_geometric_pose_estimation(synthetic_observation):
    observation, truth = synthetic_observation

    estimator = GeometricPoseEstimator(
        minimum_points=6,
        reprojection_threshold_px=5.0,
    )

    config = EstimatorConfig(name="geometric")

    estimator.initialize(config, observation)

    result = estimator.step(observation, 0.02)

    assert result.pose.frame_from == "L"
    assert result.pose.frame_to == "F"

    assert result.diagnostics["measurement_success"]

    position_error = np.linalg.norm(
        result.pose.position - truth.position
    )

    assert position_error < 2.0


def test_geometric_reprojection(synthetic_observation):
    observation, truth = synthetic_observation

    error = pose_reprojection_error(
        observation,
        truth,
    )

    assert error is not None
    assert error < 1e-6


def test_geometric_failure_with_missing_features(
    synthetic_observation,
):
    observation, truth = synthetic_observation

    mask = np.zeros(14, dtype=bool)
    mask[:3] = True

    limited = CameraObservation(
        timestamp=0.0,
        frame_index=0,
        keypoints=observation.keypoints,
        visible=mask,
        camera=observation.camera,
        camera_pose=observation.camera_pose,
    )

    estimator = GeometricPoseEstimator(
        minimum_points=6
    )

    estimator.initialize(
        EstimatorConfig(name="geometric"),
        limited,
    )

    with pytest.raises(RuntimeError):
        estimator.step(limited, 0.02)


def test_hybrid_estimator(synthetic_observation):
    observation, truth = synthetic_observation

    improved, config = make_improved_filter(truth)

    geometric = GeometricPoseEstimator(
        minimum_points=6
    )

    hybrid = HybridPoseEstimator(
        improved,
        geometric,
        maximum_geometric_error_px=4.0,
        minimum_geometric_inliers=6,
    )

    hybrid.initialize(config, observation)

    result = hybrid.step(observation, 0.02)

    assert result.estimator_name == "hybrid"

    assert result.diagnostics[
        "selected_estimator"
    ] in ("geometric", "improved_pf")

    assert np.isfinite(result.pose.position).all()

    hybrid.reset()


def test_reproducibility(synthetic_observation):
    observation, truth = synthetic_observation

    estimates = []

    for _ in range(2):
        estimator, config = make_improved_filter(
            truth, seed=42
        )

        estimator.initialize(config, observation)

        positions = []

        for step in range(4):
            result = estimator.step(
                observation,
                dt=0.02,
            )
            positions.append(result.pose.position.copy())

        estimates.append(np.asarray(positions))

    np.testing.assert_allclose(
        estimates[0],
        estimates[1],
        atol=1e-12,
        rtol=0,
    )


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ImprovedParticleFilter(
            ann_predictor=lambda obs: 0,
            calibration_errors=np.zeros((5, 4)),
            measurement_sigma_px=-1,
        )

    with pytest.raises(ValueError):
        GeometricPoseEstimator(
            reprojection_threshold_px=-1
        )

    with pytest.raises(ValueError):
        HybridPoseEstimator(
            particle_filter=object(),
            geometric_estimator=object(),
            minimum_geometric_inliers=2,
        )
