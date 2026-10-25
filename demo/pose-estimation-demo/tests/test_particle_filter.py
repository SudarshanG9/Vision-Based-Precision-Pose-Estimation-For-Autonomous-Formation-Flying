
"""Consolidated Phase 4 Particle Filter tests.

Run:
    python -m pytest tests/test_particle_filter.py -v
"""

import numpy as np
import pytest

from estimators.interfaces import EstimatorConfig
from estimators.particle_filter_paper import (
    PaperParticleFilter,
    quaternion_products,
    quaternion_matrices,
    small_angle_quaternions,
)
from models.labels import encode_pose
from simulation.scenarios import generate_scenario
from simulation.transforms import quaternion_to_euler
from scripts.run_experiment import (
    generate_baseline_experiment,
    run_filter,
)


def make_calibration_errors(seed=42, n=200):
    rng = np.random.default_rng(seed)

    return np.column_stack([
        rng.normal(0.0, 2.0, n),
        rng.normal(0.0, 1.5, n),
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, np.deg2rad(3.0), n),
    ])


def make_oracle_predictor(truth):
    """
    Test-only classifier based on known ground truth.

    Never use this predictor in research benchmarks.
    """
    def predict(observation):
        pose = truth[observation.frame_index]

        roll = quaternion_to_euler(
            pose.quaternion
        )[0]

        target = [
            pose.position[0],
            pose.position[1],
            pose.position[2],
            roll,
        ]

        return encode_pose(target)

    return predict


@pytest.fixture
def short_experiment():
    observations, truth = generate_baseline_experiment(
        seed=42,
        duration=0.5,
        fps=20,
        pixel_noise_std=1.0,
    )

    return observations, truth


def make_filter(truth, n=128, seed=42, interval=1):
    estimator = PaperParticleFilter(
        ann_predictor=make_oracle_predictor(truth),
        calibration_errors=make_calibration_errors(),
        reseed_interval=interval,
        particle_sample_limit=20,
    )

    config = EstimatorConfig(
        name="paper_pf",
        particle_count=n,
        resampling_alpha=0.9,
        position_noise_std=0.2,
        rotation_noise_std=0.01,
        seed=seed,
    )

    return estimator, config


def test_quaternion_operations():
    identity = np.array([[0, 0, 0, 1.0]])

    result = quaternion_products(identity, identity)

    np.testing.assert_allclose(
        result, identity, atol=1e-12
    )

    rotations = quaternion_matrices(identity)

    np.testing.assert_allclose(
        rotations[0], np.eye(3), atol=1e-12
    )


def test_small_angle_quaternions():
    vectors = np.zeros((10, 3))

    quaternions = small_angle_quaternions(vectors)

    np.testing.assert_allclose(
        quaternions,
        np.tile([0, 0, 0, 1], (10, 1)),
    )


def test_filter_initialization(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth, n=128)

    estimator.initialize(config, observations[0])

    assert estimator.particles.shape == (128, 7)
    assert estimator.weights.shape == (128,)

    np.testing.assert_allclose(
        estimator.weights.sum(), 1.0
    )

    np.testing.assert_allclose(
        np.linalg.norm(estimator.particles[:, 3:], axis=1),
        1.0,
        atol=1e-10,
    )


def test_projection_shape(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth, n=64)
    estimator.initialize(config, observations[0])

    projected, valid = estimator._project_particles(
        observations[0]
    )

    assert projected.shape == (64, 14, 2)
    assert valid.shape == (64, 14)
    assert valid.dtype == np.bool_


def test_measurement_weights(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth, n=64)
    estimator.initialize(config, observations[0])

    mse, count = estimator._measurement_update(
        observations[0]
    )

    assert count >= 0
    assert count <= 14

    if count > 0:
        assert mse is None or mse >= 0

    assert np.isfinite(estimator.weights).all()

    np.testing.assert_allclose(
        estimator.weights.sum(),
        1.0,
        atol=1e-10,
    )


def test_resampling_population(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth, n=101)
    estimator.initialize(config, observations[0])

    estimator._resample(
        observations[0],
        ann_reseed=True,
    )

    assert estimator.particles.shape == (101, 7)

    np.testing.assert_allclose(
        estimator.weights,
        np.full(101, 1 / 101),
    )


def test_filter_step(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth, n=128)
    estimator.initialize(config, observations[0])

    result = estimator.step(
        observations[0],
        dt=0.05,
    )

    assert result.estimator_name == "paper_pf"
    assert result.pose.position.shape == (3,)
    assert result.pose.quaternion.shape == (4,)

    assert result.runtime_ms >= 0

    diagnostics = result.diagnostics

    assert 1 <= diagnostics["effective_sample_size"] <= 128.001
    assert diagnostics["particle_count"] == 128

    assert len(diagnostics["particle_sample"]) == 20


def test_resampling_schedule(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(
        truth, n=64, interval=3
    )

    estimator.initialize(config, observations[0])

    flags = []

    for observation in observations[:7]:
        result = estimator.step(observation, dt=0.05)
        flags.append(
            result.diagnostics["ann_reseed_performed"]
        )

    assert flags == [
        True, False, False, True,
        False, False, True,
    ]


def test_reproducible_filter(short_experiment):
    observations, truth = short_experiment

    histories = []

    for _ in range(2):
        estimator, config = make_filter(
            truth, n=96, seed=42
        )

        estimator.initialize(config, observations[0])

        estimates = []

        for observation in observations:
            result = estimator.step(
                observation, dt=0.05
            )

            estimates.append(
                result.pose.position.copy()
            )

        histories.append(np.array(estimates))

    np.testing.assert_allclose(
        histories[0], histories[1],
        rtol=0,
        atol=1e-12,
    )


def test_complete_experiment(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(
        truth, n=96
    )

    result = run_filter(
        estimator,
        config,
        observations,
        truth,
        fps=20,
    )

    assert result["summary"]["frame_count"] == 10
    assert len(result["frames"]) == 10

    assert np.isfinite(
        result["summary"]["mean_position_error_m"]
    )

    assert np.isfinite(
        result["summary"]["mean_attitude_error_deg"]
    )


def test_no_ground_truth_required_by_filter(short_experiment):
    observations, truth = short_experiment

    # Only the synthetic ANN test stub references ground truth.
    # The Particle Filter itself receives observations only.
    estimator, config = make_filter(truth, n=64)

    estimator.initialize(config, observations[0])

    result = estimator.step(
        observations[0], dt=0.05
    )

    assert np.isfinite(result.pose.position).all()
    assert np.isfinite(result.pose.quaternion).all()


def test_filter_reset(short_experiment):
    observations, truth = short_experiment

    estimator, config = make_filter(truth)
    estimator.initialize(config, observations[0])

    estimator.reset()

    assert estimator.particles is None
    assert estimator.weights is None

    with pytest.raises(RuntimeError):
        estimator.step(observations[0], 0.05)


def test_invalid_calibration():
    with pytest.raises(ValueError):
        PaperParticleFilter(
            ann_predictor=lambda obs: 0,
            calibration_errors=np.zeros((10, 3)),
        )
