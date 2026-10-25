
"""Generate reproducible synthetic ANN classification datasets.

Run from pose-estimation-demo/:

    python -m training.generate_dataset \
        --config configs/training.yaml

The generator creates trajectory-disjoint train/val/test splits.

Ground-truth pose labels are saved as targets only.
They are not included in the network's input features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from estimators.interfaces import Pose3D
from models.labels import DEFAULT_DISCRETIZATION
from models.preprocessing import (
    preprocess_observation,
    validate_features,
)
from simulation.aircraft_model import create_aircraft_model
from simulation.camera import default_camera
from simulation.trajectory import (
    TrajectoryConfig,
    generate_trajectory,
)
from simulation.renderer import render_aircraft
from simulation.observations import (
    ObservationConfig,
    generate_observation,
)
from simulation.transforms import (
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
)


@dataclass
class DatasetConfig:
    seed: int = 42

    num_trajectories: int = 300
    duration_seconds: float = 10.0
    simulation_fps: int = 20

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    pixel_noise_std: float = 1.5
    dropout_probability: float = 0.10

    output_directory: str = "data/processed"

    def __post_init__(self):
        if self.num_trajectories < 3:
            raise ValueError(
                "At least three trajectories are required"
            )

        if self.duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be positive"
            )

        if self.simulation_fps < 2:
            raise ValueError(
                "simulation_fps must be at least 2"
            )

        fractions = np.array([
            self.train_fraction,
            self.val_fraction,
            self.test_fraction,
        ])

        if np.any(fractions <= 0):
            raise ValueError(
                "All split fractions must be positive"
            )

        if not np.isclose(fractions.sum(), 1.0):
            raise ValueError(
                "Split fractions must sum to 1"
            )

        if self.pixel_noise_std < 0:
            raise ValueError(
                "pixel_noise_std must be nonnegative"
            )

        if not 0 <= self.dropout_probability <= 1:
            raise ValueError(
                "dropout_probability must be in [0, 1]"
            )


def load_config(path: str | Path) -> DatasetConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a map")

    section = raw.get("dataset")

    if not isinstance(section, dict):
        raise ValueError("Missing dataset configuration")

    return DatasetConfig(**section)


def split_trajectory_ids(
    count: int,
    fractions: tuple[float, float, float],
    seed: int,
):
    """Split entire trajectories, never individual frames."""
    if count < 3:
        raise ValueError("Need at least three trajectories")

    rng = np.random.default_rng(seed)
    ids = rng.permutation(count)

    raw = np.array(fractions) * count
    sizes = np.floor(raw).astype(int)

    remainder = count - int(sizes.sum())

    order = np.argsort(-(raw - sizes))

    for index in order[:remainder]:
        sizes[index] += 1

    # Guarantee nonempty partitions.
    for index in range(3):
        if sizes[index] == 0:
            donor = int(np.argmax(sizes))
            if sizes[donor] <= 1:
                raise ValueError("Cannot construct nonempty splits")
            sizes[donor] -= 1
            sizes[index] += 1

    train_end = sizes[0]
    val_end = train_end + sizes[1]

    return {
        "train": ids[:train_end].tolist(),
        "val": ids[train_end:val_end].tolist(),
        "test": ids[val_end:].tolist(),
    }


def sample_trajectory_variation(rng):
    """Sample variations without exposing test poses to training."""
    return {
        "start_distance": float(rng.uniform(39.0, 48.0)),
        "end_distance": float(rng.uniform(20.0, 29.0)),
        "lateral_shift": float(rng.uniform(-17.0, 17.0)),
        "vertical_shift": float(rng.uniform(-6.0, 6.0)),
        "roll_shift": float(np.deg2rad(
            rng.uniform(-12.0, 12.0)
        )),
        "pitch_shift": float(np.deg2rad(
            rng.uniform(-5.0, 5.0)
        )),
        "yaw_shift": float(np.deg2rad(
            rng.uniform(-5.0, 5.0)
        )),
        "turbulence": float(rng.uniform(0.0, 0.6)),
        "aggressive_bank": bool(rng.random() < 0.25),
    }


def vary_pose(
    original: Pose3D,
    timestamp: float,
    duration: float,
    variation: dict,
):
    """Generate a consistent trajectory-specific pose variation."""
    tau = np.clip(timestamp / duration, 0.0, 1.0)

    start = variation["start_distance"]
    end = variation["end_distance"]

    # All generated samples remain within the paper label bounds.
    position = original.position.copy()
    position[0] = start + (end - start) * tau

    position[1] += variation["lateral_shift"]
    position[2] += variation["vertical_shift"]

    offset_rotation = euler_to_quaternion(
        variation["roll_shift"],
        variation["pitch_shift"],
        variation["yaw_shift"],
    )

    quaternion = quaternion_multiply(
        offset_rotation,
        original.quaternion,
    )

    return Pose3D(
        position=position,
        quaternion=quaternion,
        frame_from="L",
        frame_to="F",
    )


def pose_to_label_vector(pose: Pose3D):
    """Convert full pose to [x, y, z, roll]."""
    roll = quaternion_to_euler(
        pose.quaternion
    )[0]

    return np.array([
        pose.position[0],
        pose.position[1],
        pose.position[2],
        roll,
    ], dtype=np.float64)


def generate_one_trajectory(
    trajectory_id: int,
    config: DatasetConfig,
):
    """Generate ML examples for a single independent trajectory."""

    trajectory_seed = config.seed + 104729 * (
        trajectory_id + 1
    )

    rng = np.random.default_rng(trajectory_seed)

    variation = sample_trajectory_variation(rng)

    base_config = TrajectoryConfig(
        duration=config.duration_seconds,
        fps=config.simulation_fps,
        initial_distance=80.0,
        final_distance=35.0,
        turbulence=variation["turbulence"],
        aggressive_bank=variation["aggressive_bank"],
    )

    trajectory = generate_trajectory(
        base_config,
        seed=trajectory_seed,
    )

    aircraft = create_aircraft_model()
    camera = default_camera()

    observation_config = ObservationConfig(
        pixel_noise_std=config.pixel_noise_std,
        dropout_probability=config.dropout_probability,
    )

    features = []
    labels = []
    poses = []
    frame_indices = []
    visible_counts = []

    for frame_index, timestamp in enumerate(
        trajectory.timestamps
    ):
        pose = vary_pose(
            original=trajectory.relative_poses[frame_index],
            timestamp=float(timestamp),
            duration=config.duration_seconds,
            variation=variation,
        )

        target = pose_to_label_vector(pose)

        if not DEFAULT_DISCRETIZATION.validate(target)[0]:
            raise ValueError(
                f"Generated pose is outside the paper's "
                f"label space: trajectory={trajectory_id}, "
                f"frame={frame_index}, pose={target}"
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
            frame_index=frame_index,
            rng=rng,
            config=observation_config,
        )

        features.append(
            preprocess_observation(observation)
        )

        labels.append(
            DEFAULT_DISCRETIZATION.encode(target)
        )

        poses.append(target)
        frame_indices.append(frame_index)
        visible_counts.append(observation.visible_count)

    return {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "pose_4d": np.asarray(poses, dtype=np.float64),
        "trajectory_id": np.full(
            len(labels),
            trajectory_id,
            dtype=np.int32,
        ),
        "frame_index": np.asarray(
            frame_indices,
            dtype=np.int32,
        ),
        "visible_count": np.asarray(
            visible_counts,
            dtype=np.int16,
        ),
    }


def concatenate_records(records):
    if not records:
        raise ValueError("No trajectory records supplied")

    keys = records[0].keys()

    return {
        key: np.concatenate([
            record[key] for record in records
        ], axis=0)
        for key in keys
    }


def build_dataset(config: DatasetConfig):
    """Build the full dataset in memory, separated by trajectories."""

    split_ids = split_trajectory_ids(
        count=config.num_trajectories,
        fractions=(
            config.train_fraction,
            config.val_fraction,
            config.test_fraction,
        ),
        seed=config.seed,
    )

    trajectory_to_split = {
        trajectory_id: split
        for split, ids in split_ids.items()
        for trajectory_id in ids
    }

    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for trajectory_id in range(config.num_trajectories):
        record = generate_one_trajectory(
            trajectory_id,
            config,
        )

        destination = trajectory_to_split[trajectory_id]
        records[destination].append(record)

    datasets = {
        split: concatenate_records(split_records)
        for split, split_records in records.items()
    }

    for data in datasets.values():
        validate_features(data["X"])

        if np.any(data["y"] < 0) or np.any(
            data["y"] >= DEFAULT_DISCRETIZATION.num_classes
        ):
            raise ValueError("Invalid label detected")

    return datasets, split_ids


def save_dataset(
    config: DatasetConfig,
    datasets: dict,
    split_ids: dict,
):
    """Save compressed arrays and reproducibility metadata."""

    directory = Path(config.output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0",
        "dataset_config": asdict(config),
        "label_shape": list(
            DEFAULT_DISCRETIZATION.shape
        ),
        "num_classes": DEFAULT_DISCRETIZATION.num_classes,
        "feature_dimension": 42,
        "landmark_count": 14,
        "splits": {},
    }

    for split, data in datasets.items():
        path = directory / f"{split}.npz"

        np.savez_compressed(path, **data)

        manifest["splits"][split] = {
            "filename": path.name,
            "samples": int(len(data["y"])),
            "trajectory_ids": split_ids[split],
            "unique_classes": int(
                len(np.unique(data["y"]))
            ),
            "mean_visible_landmarks": float(
                np.mean(data["visible_count"])
            ),
        }

    manifest_path = directory / "manifest.json"

    with manifest_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            allow_nan=False,
        )

    return manifest


def load_dataset(
    directory: str | Path,
    split: str,
):
    """Load one split for downstream ANN training."""

    if split not in ("train", "val", "test"):
        raise ValueError("Unknown dataset split")

    path = Path(directory) / f"{split}.npz"

    with np.load(path, allow_pickle=False) as data:
        result = {
            key: data[key].copy()
            for key in data.files
        }

    validate_features(result["X"])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic pose classification datasets"
    )

    parser.add_argument(
        "--config",
        default="configs/training.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    datasets, split_ids = build_dataset(config)

    manifest = save_dataset(
        config,
        datasets,
        split_ids,
    )

    print("\nSynthetic dataset generation complete")
    print("-" * 48)

    for split, details in manifest["splits"].items():
        print(
            f"{split:>5}: "
            f"{details['samples']:>7} samples | "
            f"{details['unique_classes']:>4} classes | "
            f"{len(details['trajectory_ids']):>4} trajectories"
        )

    print("-" * 48)
    print(f"Output: {config.output_directory}")


if __name__ == "__main__":
    main()
