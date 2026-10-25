
"""Complete synthetic scenario generation and evaluation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.interfaces import CameraObservation, Pose3D
from simulation.aircraft_model import (
    AircraftModel,
    create_aircraft_model,
)
from simulation.camera import PinholeCamera, default_camera
from simulation.trajectory import (
    Trajectory,
    TrajectoryConfig,
    generate_trajectory,
)
from simulation.renderer import render_aircraft
from simulation.observations import (
    ObservationConfig,
    generate_observation,
)


@dataclass
class ScenarioData:

    scenario_id: str
    seed: int

    aircraft: AircraftModel
    camera: PinholeCamera

    trajectory: Trajectory

    observations: list[CameraObservation]

    def __post_init__(self):
        if len(self.observations) != len(
            self.trajectory.timestamps
        ):
            raise ValueError(
                "Observations and trajectory lengths differ"
            )

    @property
    def frame_count(self):
        return len(self.observations)

    @property
    def ground_truth(self) -> list[Pose3D]:
        """
        Evaluator-only ground truth.

        Do not pass this property to estimator methods.
        """
        return self.trajectory.relative_poses


SCENARIO_NAMES = (
    "Approach_L1",
    "Approach_L2",
    "High_Turbulence",
    "Aggressive_Banking",
    "High_Occlusion",
    "Low_Visibility",
    "Noise_Stress",
)


def scenario_configuration(
    scenario_id: str,
    duration: float = 10.0,
    fps: int = 100,
):
    """Return trajectory and observation configurations."""

    if scenario_id not in SCENARIO_NAMES:
        raise ValueError(
            f"Unknown scenario: {scenario_id}. "
            f"Available: {SCENARIO_NAMES}"
        )

    trajectory = TrajectoryConfig(
        duration=duration,
        fps=fps,
    )

    observation = ObservationConfig(
        pixel_noise_std=1.5,
        dropout_probability=0.0,
    )

    if scenario_id == "Approach_L2":
        trajectory.lateral_offset = 5.0
        trajectory.vertical_offset = -2.0

    elif scenario_id == "High_Turbulence":
        trajectory.turbulence = 1.0

    elif scenario_id == "Aggressive_Banking":
        trajectory.aggressive_bank = True

    elif scenario_id == "High_Occlusion":
        observation.dropout_probability = 0.35

    elif scenario_id == "Low_Visibility":
        observation.dropout_probability = 0.60
        observation.pixel_noise_std = 3.0

    elif scenario_id == "Noise_Stress":
        observation.pixel_noise_std = 7.0

    return trajectory, observation


def generate_scenario(
    scenario_id: str = "Approach_L1",
    seed: int = 42,
    duration: float = 10.0,
    fps: int = 100,
    camera: PinholeCamera | None = None,
) -> ScenarioData:
    """Generate an entire experiment-ready synthetic trajectory."""

    aircraft = create_aircraft_model()

    if camera is None:
        camera = default_camera()

    trajectory_config, observation_config = (
        scenario_configuration(
            scenario_id,
            duration=duration,
            fps=fps,
        )
    )

    trajectory = generate_trajectory(
        trajectory_config,
        seed=seed,
    )

    # Separate measurement RNG from trajectory RNG.
    rng = np.random.default_rng(seed + 100_003)

    observations = []

    for frame_index, (timestamp, relative_pose) in enumerate(
        zip(
            trajectory.timestamps,
            trajectory.relative_poses,
        )
    ):
        rendered = render_aircraft(
            aircraft=aircraft,
            camera=camera,
            leader_in_follower=relative_pose,
        )

        observation = generate_observation(
            render=rendered,
            camera=camera,
            timestamp=float(timestamp),
            frame_index=frame_index,
            rng=rng,
            config=observation_config,
        )

        observations.append(observation)

    return ScenarioData(
        scenario_id=scenario_id,
        seed=seed,
        aircraft=aircraft,
        camera=camera,
        trajectory=trajectory,
        observations=observations,
    )


def scenario_summary(data: ScenarioData):
    """Return numerical summary statistics for verification."""

    visible_counts = np.array([
        obs.visible_count
        for obs in data.observations
    ])

    positions = np.array([
        pose.position
        for pose in data.ground_truth
    ])

    distances = np.linalg.norm(positions, axis=1)

    return {
        "scenario_id": data.scenario_id,
        "seed": data.seed,
        "frame_count": data.frame_count,
        "duration_seconds": float(
            data.trajectory.timestamps[-1]
            - data.trajectory.timestamps[0]
        ),
        "mean_visible_features": float(
            np.mean(visible_counts)
        ),
        "minimum_visible_features": int(
            np.min(visible_counts)
        ),
        "initial_distance_m": float(distances[0]),
        "final_distance_m": float(distances[-1]),
    }
