
"""Run a paper-baseline ANN + Particle Filter experiment.

Usage:
    python -m scripts.run_experiment \
        --config configs/paper_baseline.yaml \
        --variant paper_1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from estimators.interfaces import (
    EstimatorConfig,
    Pose3D,
)

from estimators.particle_filter_paper import PaperParticleFilter
from models.preprocessing import preprocess_observation
from simulation.aircraft_model import create_aircraft_model
from simulation.camera import default_camera
from simulation.renderer import render_aircraft
from simulation.observations import (
    ObservationConfig,
    generate_observation,
)
from simulation.trajectory import (
    TrajectoryConfig,
    generate_trajectory,
)
from simulation.transforms import (
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    rotation_error_degrees,
)
from training.train_classifier import (
    load_classifier,
    resolve_device,
)


def make_ann_predictor(model, device):
    """Wrap a trained ANN into the PF's observation interface."""

    def predict(observation):
        features = preprocess_observation(observation)

        tensor = torch.from_numpy(
            features[None, :]
        ).to(device)

        model.eval()

        with torch.inference_mode():
            logits = model(tensor)
            return int(logits.argmax(dim=1).item())

    return predict


def generate_baseline_experiment(
    *,
    seed: int,
    duration: float,
    fps: int,
    pixel_noise_std: float,
):
    """
    Construct an approach trajectory compatible with the
    Phase 2 classifier training distribution.

    This is intentionally not the unmodified Phase 1 Approach_L1,
    whose initial position exceeds the classifier's label range.
    """
    rng = np.random.default_rng(seed + 100003)

    trajectory = generate_trajectory(
        TrajectoryConfig(
            duration=duration,
            fps=fps,
            initial_distance=80.0,
            final_distance=35.0,
        ),
        seed=seed,
    )

    aircraft = create_aircraft_model()
    camera = default_camera()

    observations = []
    ground_truth = []

    observation_config = ObservationConfig(
        pixel_noise_std=pixel_noise_std,
        dropout_probability=0.10,
    )

    for index, timestamp in enumerate(trajectory.timestamps):
        tau = float(timestamp / duration)

        original = trajectory.relative_poses[index]

        position = original.position.copy()
        position[0] = 46.0 - 21.0 * tau

        # Preserve smooth attitude motion.
        pose = Pose3D(
            position=position,
            quaternion=original.quaternion,
            frame_from="L",
            frame_to="F",
        )

        rendered = render_aircraft(
            aircraft,
            camera,
            pose,
        )

        observation = generate_observation(
            render=rendered,
            camera=camera,
            timestamp=float(timestamp),
            frame_index=index,
            rng=rng,
            config=observation_config,
        )

        ground_truth.append(pose)
        observations.append(observation)

    return observations, ground_truth


def run_filter(
    filter_instance,
    config,
    observations,
    ground_truth,
    *,
    fps,
):
    """Evaluate one estimator using the generated observations."""
    filter_instance.initialize(config, observations[0])

    telemetry = []

    position_errors = []
    attitude_errors = []

    dt = 1.0 / fps

    for observation, truth in zip(
        observations, ground_truth
    ):
        result = filter_instance.step(observation, dt)

        position_error = float(
            np.linalg.norm(
                result.pose.position - truth.position
            )
        )

        attitude_error = rotation_error_degrees(
            truth,
            result.pose,
        )

        position_errors.append(position_error)
        attitude_errors.append(attitude_error)

        telemetry.append({
            "frame_index": observation.frame_index,
            "timestamp": observation.timestamp,
            "ground_truth": {
                "position": truth.position.tolist(),
                "quaternion": truth.quaternion.tolist(),
            },
            "estimate": {
                "position": result.pose.position.tolist(),
                "quaternion": result.pose.quaternion.tolist(),
            },
            "position_error_m": position_error,
            "attitude_error_deg": attitude_error,
            "runtime_ms": result.runtime_ms,
            "diagnostics": result.diagnostics,
        })

    filter_instance.reset()

    errors = np.asarray(position_errors)

    return {
        "summary": {
            "mean_position_error_m": float(errors.mean()),
            "median_position_error_m": float(np.median(errors)),
            "p95_position_error_m": float(
                np.percentile(errors, 95)
            ),
            "mean_attitude_error_deg": float(
                np.mean(attitude_errors)
            ),
            "p95_attitude_error_deg": float(
                np.percentile(attitude_errors, 95)
            ),
            "frame_count": len(telemetry),
        },
        "frames": telemetry,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/paper_baseline.yaml",
    )
    parser.add_argument(
        "--variant",
        choices=["paper_1000", "paper_5000"],
        default="paper_1000",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    device = resolve_device(args.device)

    checkpoints = settings["checkpoints"]

    model, _ = load_classifier(
        checkpoints["classifier"],
        device,
    )

    with np.load(
        checkpoints["calibration"],
        allow_pickle=False,
    ) as calibration:
        calibration_errors = calibration["errors"].copy()

    predictor = make_ann_predictor(model, device)

    experiment = settings["experiment"]
    filter_settings = settings["particle_filter"]
    variant = settings["configurations"][args.variant]

    observations, truth = generate_baseline_experiment(
        seed=int(experiment["seed"]),
        duration=float(experiment["duration_seconds"]),
        fps=int(experiment["simulation_fps"]),
        pixel_noise_std=float(
            settings["camera"]["pixel_noise_std"]
        ),
    )

    estimator = PaperParticleFilter(
        ann_predictor=predictor,
        calibration_errors=calibration_errors,
        reseed_interval=int(variant["reseed_interval"]),
        likelihood_epsilon=float(
            filter_settings["likelihood_epsilon"]
        ),
        particle_sample_limit=int(
            filter_settings["particle_sample_limit"]
        ),
    )

    estimator_config = EstimatorConfig(
        name="paper_pf",
        particle_count=int(variant["particle_count"]),
        resampling_alpha=float(
            variant["resampling_alpha"]
        ),
        position_noise_std=float(
            filter_settings["position_noise_std"]
        ),
        rotation_noise_std=float(
            filter_settings["rotation_noise_std"]
        ),
        seed=int(experiment["seed"]),
    )

    result = run_filter(
        estimator,
        estimator_config,
        observations,
        truth,
        fps=int(experiment["simulation_fps"]),
    )

    result["metadata"] = {
        "variant": args.variant,
        "scenario": "distribution_compatible_approach",
        "seed": int(experiment["seed"]),
        "particle_count": estimator_config.particle_count,
        "alpha": estimator_config.resampling_alpha,
        "reseed_interval": estimator.reseed_interval,
    }

    output_directory = Path(
        settings["outputs"]["directory"]
    )
    output_directory.mkdir(
        parents=True, exist_ok=True
    )

    output_path = output_directory / (
        f"{args.variant}_seed{experiment['seed']}.json"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            indent=2,
            allow_nan=False,
        )

    print("\nPaper Particle Filter experiment")
    print("-" * 52)

    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
