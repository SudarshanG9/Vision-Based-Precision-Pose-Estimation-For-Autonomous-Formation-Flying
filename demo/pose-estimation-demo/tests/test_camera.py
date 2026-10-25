
"""Consolidated Phase 1 simulation validation."""

import json

import numpy as np
import pytest

from estimators.interfaces import (
    Pose3D,
    CameraObservation,
    ANN_INPUT_DIM,
    NUM_LANDMARKS,
)

from simulation.transforms import (
    euler_to_quaternion,
    compose_poses,
    relative_pose,
    rotation_error_degrees,
)

from simulation.aircraft_model import create_aircraft_model
from simulation.camera import default_camera
from simulation.trajectory import (
    TrajectoryConfig,
    generate_trajectory,
)
from simulation.renderer import (
    render_aircraft,
    ray_triangle_intersection,
)
from simulation.observations import observation_to_json
from simulation.scenarios import (
    generate_scenario,
    scenario_summary,
    SCENARIO_NAMES,
)


def test_aircraft_geometry():
    aircraft = create_aircraft_model()

    assert aircraft.landmarks.shape == (14, 3)
    assert len(aircraft.names) == 14
    assert len(set(aircraft.names)) == 14

    assert np.all(np.isfinite(aircraft.landmarks))
    assert np.all(aircraft.bounding_extent() > 0)


def test_transformation_composition():
    T_W_F = Pose3D(
        position=[10, 2, -1],
        quaternion=euler_to_quaternion(
            0.1, -0.2, 0.3
        ),
        frame_from="F",
        frame_to="W",
    )

    T_F_L = Pose3D(
        position=[50, 3, -2],
        quaternion=euler_to_quaternion(
            0.2, 0.1, 0.0
        ),
        frame_from="L",
        frame_to="F",
    )

    T_W_L = compose_poses(T_W_F, T_F_L)

    recovered = relative_pose(T_W_F, T_W_L)

    np.testing.assert_allclose(
        recovered.as_matrix(),
        T_F_L.as_matrix(),
        atol=1e-10,
    )


def test_camera_mount_and_projection():
    camera = default_camera()

    # A point 50 m in front of the follower.
    leader_pose = Pose3D(
        position=[50, 0, 0],
        quaternion=[0, 0, 0, 1],
        frame_from="L",
        frame_to="F",
    )

    pixels, valid, _ = camera.project_landmarks(
        np.array([[0.0, 0.0, 0.0]]),
        leader_pose,
    )

    assert valid[0]

    np.testing.assert_allclose(
        pixels[0],
        [320.0, 240.0],
        atol=1e-8,
    )


def test_projection_depth_rejection():
    camera = default_camera()

    points = np.array([
        [0, 0, 10],
        [0, 0, -10],
        [100, 0, 10],
    ], dtype=float)

    pixels, valid = camera.project_camera_points(points)

    assert valid.tolist() == [True, False, False]

    np.testing.assert_allclose(
        pixels[0],
        [320, 240],
    )

    assert np.isnan(pixels[1]).all()


def test_ray_triangle_occlusion():
    triangle = np.array([
        [-1, -1, 5],
        [1, -1, 5],
        [0, 1, 5],
    ], dtype=float)

    assert ray_triangle_intersection(
        np.array([0, 0, 10]),
        triangle,
    )

    assert not ray_triangle_intersection(
        np.array([0, 0, 3]),
        triangle,
    )


def test_rendered_aircraft():
    aircraft = create_aircraft_model()
    camera = default_camera()

    pose = Pose3D(
        position=[50, 0, 0],
        quaternion=[0, 0, 0, 1],
        frame_from="L",
        frame_to="F",
    )

    result = render_aircraft(
        aircraft,
        camera,
        pose,
    )

    assert result.image.shape == (480, 640, 3)
    assert result.image.dtype == np.uint8
    assert result.pixels.shape == (14, 2)
    assert result.visible.shape == (14,)
    assert result.camera_points.shape == (14, 3)

    assert np.count_nonzero(result.visible) > 0


def test_trajectory_generation():
    config = TrajectoryConfig(
        duration=10.0,
        fps=100,
    )

    trajectory = generate_trajectory(config, seed=42)

    assert len(trajectory.timestamps) == 1000
    assert len(trajectory.relative_poses) == 1000

    start = trajectory.relative_poses[0].position[0]
    end = trajectory.relative_poses[-1].position[0]

    assert start > end

    assert np.all(np.isfinite([
        pose.position
        for pose in trajectory.relative_poses
    ]))


def test_rotation_error():
    truth = Pose3D.identity("F")

    estimate = Pose3D(
        position=[0, 0, 0],
        quaternion=euler_to_quaternion(
            np.deg2rad(30), 0, 0
        ),
        frame_from="F",
        frame_to="F",
    )

    error = rotation_error_degrees(truth, estimate)

    assert error == pytest.approx(30.0, abs=1e-8)


def test_nominal_scenario():
    data = generate_scenario(
        scenario_id="Approach_L1",
        seed=42,
        duration=1.0,
        fps=20,
    )

    assert data.frame_count == 20

    assert len(data.observations) == 20
    assert len(data.ground_truth) == 20

    for observation in data.observations:
        assert isinstance(observation, CameraObservation)
        assert observation.keypoints.shape == (
            NUM_LANDMARKS, 2
        )

        features = observation.ann_input()

        assert features.shape == (ANN_INPUT_DIM,)
        assert np.all(np.isfinite(features))


def test_scenario_reproducibility():
    first = generate_scenario(
        "High_Turbulence",
        seed=42,
        duration=1.0,
        fps=20,
    )

    second = generate_scenario(
        "High_Turbulence",
        seed=42,
        duration=1.0,
        fps=20,
    )

    for obs_a, obs_b in zip(
        first.observations,
        second.observations,
    ):
        np.testing.assert_allclose(
            obs_a.keypoints,
            obs_b.keypoints,
            equal_nan=True,
        )

        np.testing.assert_array_equal(
            obs_a.visible,
            obs_b.visible,
        )

    for pose_a, pose_b in zip(
        first.ground_truth,
        second.ground_truth,
    ):
        np.testing.assert_allclose(
            pose_a.as_matrix(),
            pose_b.as_matrix(),
        )


def test_scenarios_and_noise():
    for scenario_id in SCENARIO_NAMES:
        data = generate_scenario(
            scenario_id,
            seed=123,
            duration=1.0,
            fps=20,
        )

        assert data.frame_count == 20

        summary = scenario_summary(data)

        assert summary["frame_count"] == 20
        assert summary["initial_distance_m"] > (
            summary["final_distance_m"]
        )


def test_json_observation():
    data = generate_scenario(
        "Approach_L1",
        duration=1.0,
        fps=20,
    )

    record = observation_to_json(
        data.observations[0]
    )

    assert len(record["keypoints"]) == 14
    assert len(record["visible"]) == 14

    # Serialization must reject no non-standard NaN values.
    serialized = json.dumps(record, allow_nan=False)

    assert isinstance(serialized, str)


def test_invisible_landmarks():
    data = generate_scenario(
        "Low_Visibility",
        seed=42,
        duration=1.0,
        fps=20,
    )

    for observation in data.observations:
        assert np.isnan(
            observation.keypoints[~observation.visible]
        ).all()


def test_invalid_scenario():
    with pytest.raises(ValueError):
        generate_scenario(
            "Not_A_Real_Scenario",
            duration=1.0,
            fps=20,
        )
