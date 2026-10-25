
"""
Consolidated Phase 0 tests.

Run from pose-estimation-demo/:

    python -m pytest tests/test_transforms.py -v
"""

import numpy as np
import pytest

from estimators.interfaces import (
    NUM_LANDMARKS,
    ANN_INPUT_DIM,
    ANN_NUM_CLASSES,
    Pose3D,
    CameraIntrinsics,
    CameraObservation,
    EstimatorConfig,
    EstimatorResult,
    PoseEstimator,
)


@pytest.fixture
def phase0_camera():
    return CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


@pytest.fixture
def phase0_observation(phase0_camera):

    keypoints = np.array(
        [
            [100.0 + i * 10.0, 150.0 + i * 5.0]
            for i in range(NUM_LANDMARKS)
        ],
        dtype=np.float64,
    )

    visibility = np.ones(
        NUM_LANDMARKS, dtype=bool
    )

    visibility[3] = False
    visibility[9] = False

    return CameraObservation(
        timestamp=0.1,
        frame_index=10,
        keypoints=keypoints,
        visible=visibility,
        camera=phase0_camera,
    )


def test_phase0_constants():

    assert NUM_LANDMARKS == 14
    assert ANN_INPUT_DIM == 42
    assert ANN_NUM_CLASSES == 4800


def test_phase0_identity_pose():

    pose = Pose3D.identity("F")

    np.testing.assert_allclose(
        pose.as_matrix(),
        np.eye(4),
        atol=1e-12,
    )

    assert pose.frame_from == "F"
    assert pose.frame_to == "F"


def test_phase0_quaternion_normalization():

    pose = Pose3D(
        position=[0, 0, 0],
        quaternion=[0, 0, 0, 5],
        frame_from="L",
        frame_to="F",
    )

    np.testing.assert_allclose(
        np.linalg.norm(pose.quaternion),
        1.0,
        atol=1e-12,
    )


def test_phase0_translation():

    pose = Pose3D(
        position=[10.0, 20.0, 30.0],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        frame_from="L",
        frame_to="F",
    )

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    ])

    transformed = pose.transform_points(points)

    expected = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 22.0, 33.0],
    ])

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-12,
    )


def test_phase0_rotation():

    # 90-degree rotation about +Z.
    angle = np.pi / 2

    pose = Pose3D(
        position=[0, 0, 0],
        quaternion=[
            0.0,
            0.0,
            np.sin(angle / 2),
            np.cos(angle / 2),
        ],
        frame_from="L",
        frame_to="F",
    )

    point = np.array([[1.0, 0.0, 0.0]])

    rotated = pose.transform_points(point)

    np.testing.assert_allclose(
        rotated,
        [[0.0, 1.0, 0.0]],
        atol=1e-10,
    )


def test_phase0_inverse_transform():

    angle = np.pi / 3

    pose = Pose3D(
        position=[12.0, -5.0, 3.0],
        quaternion=[
            0.0,
            np.sin(angle / 2),
            0.0,
            np.cos(angle / 2),
        ],
        frame_from="L",
        frame_to="F",
    )

    inverse = pose.inverse()

    result = inverse.as_matrix() @ pose.as_matrix()

    np.testing.assert_allclose(
        result,
        np.eye(4),
        atol=1e-10,
    )

    assert inverse.frame_from == "F"
    assert inverse.frame_to == "L"


def test_phase0_rotation_matrix_properties():

    pose = Pose3D(
        position=[1, 2, 3],
        quaternion=[0.2, 0.3, 0.4, 0.8],
    )

    rotation = pose.rotation_matrix()

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-10,
    )

    assert np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-10,
    )


def test_phase0_invalid_quaternion():

    with pytest.raises(ValueError):
        Pose3D(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 0],
        )


def test_phase0_invalid_position():

    with pytest.raises(ValueError):
        Pose3D(
            position=[0, 0],
            quaternion=[0, 0, 0, 1],
        )


def test_phase0_camera_matrix(phase0_camera):

    expected = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        phase0_camera.matrix(),
        expected,
    )


def test_phase0_invalid_camera():

    with pytest.raises(ValueError):
        CameraIntrinsics(
            fx=-100,
            fy=800,
            cx=320,
            cy=240,
            width=640,
            height=480,
        )


def test_phase0_visibility_mask(phase0_observation):

    assert phase0_observation.visible_count == 12

    assert np.isnan(
        phase0_observation.keypoints[3]
    ).all()

    assert np.isnan(
        phase0_observation.keypoints[9]
    ).all()


def test_phase0_ann_input(phase0_observation):

    features = phase0_observation.ann_input()

    assert features.shape == (42,)

    reshaped = features.reshape(14, 3)

    assert reshaped[3, 2] == 0.0
    assert reshaped[9, 2] == 0.0

    assert reshaped[0, 2] == 1.0

    np.testing.assert_allclose(
        reshaped[3],
        [0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        reshaped[0, :2],
        [100.0 / 640.0, 150.0 / 480.0],
    )

    assert np.all(np.isfinite(features))


def test_phase0_invalid_observation(phase0_camera):

    with pytest.raises(ValueError):
        CameraObservation(
            timestamp=0.0,
            frame_index=0,
            keypoints=np.zeros((10, 2)),
            visible=np.ones(14, dtype=bool),
            camera=phase0_camera,
        )


def test_phase0_invalid_estimator_config():

    with pytest.raises(ValueError):
        EstimatorConfig(
            name="invalid",
            particle_count=1000,
            resampling_alpha=1.5,
        )


def test_phase0_estimator_result():

    pose = Pose3D.identity("F")

    result = EstimatorResult(
        timestamp=0.1,
        estimator_name="test_estimator",
        pose=pose,
        runtime_ms=1.5,
        covariance=np.eye(6),
        diagnostics={
            "effective_sample_size": 1000
        },
    )

    assert result.estimator_name == "test_estimator"
    assert result.runtime_ms == 1.5
    assert result.covariance.shape == (6, 6)


def test_phase0_invalid_covariance():

    with pytest.raises(ValueError):
        EstimatorResult(
            timestamp=0.0,
            estimator_name="invalid",
            pose=Pose3D.identity("F"),
            runtime_ms=1.0,
            covariance=-np.eye(6),
        )


def test_phase0_estimator_protocol(phase0_observation):

    class MockEstimator:

        def initialize(
            self,
            config,
            initial_observation,
        ):
            self.config = config

        def step(self, observation, dt):

            return EstimatorResult(
                timestamp=observation.timestamp,
                estimator_name="mock",
                pose=Pose3D(
                    position=[0, 0, 10],
                    quaternion=[0, 0, 0, 1],
                    frame_from="L",
                    frame_to="F",
                ),
                runtime_ms=0.1,
            )

        def reset(self):
            self.config = None

    estimator = MockEstimator()

    assert isinstance(estimator, PoseEstimator)

    config = EstimatorConfig(name="mock")

    estimator.initialize(
        config,
        phase0_observation,
    )

    result = estimator.step(
        phase0_observation,
        dt=0.01,
    )

    assert isinstance(result, EstimatorResult)

    estimator.reset()

    assert estimator.config is None
