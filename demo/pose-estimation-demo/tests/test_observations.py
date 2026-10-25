
"""Consolidated Phase 2 dataset and preprocessing tests.

Run from pose-estimation-demo/:

    python -m pytest tests/test_observations.py -v
"""

import json

import numpy as np
import pytest

from models.labels import (
    DEFAULT_DISCRETIZATION,
    encode_pose,
    decode_pose,
)

from models.preprocessing import (
    preprocess_keypoints,
    preprocess_observation,
    validate_features,
)

from training.generate_dataset import (
    DatasetConfig,
    split_trajectory_ids,
    generate_one_trajectory,
    build_dataset,
    save_dataset,
    load_dataset,
)

from simulation.scenarios import generate_scenario


def test_label_space():
    discretization = DEFAULT_DISCRETIZATION

    assert discretization.shape == (10, 10, 8, 6)
    assert discretization.num_classes == 4800

    np.testing.assert_allclose(
        discretization.bin_widths()[:3],
        [15.0, 10.0, 5.0],
    )

    assert np.isclose(
        discretization.bin_widths()[3],
        np.deg2rad(15.0),
    )


def test_label_round_trip():
    pose = np.array([
        35.0,
        2.0,
        -1.0,
        np.deg2rad(10.0),
    ])

    label = encode_pose(pose)
    decoded = decode_pose(label)

    assert 0 <= label < 4800
    assert decoded.shape == (4,)

    assert encode_pose(decoded) == label


def test_all_class_ids_round_trip():
    labels = np.arange(4800)

    decoded = DEFAULT_DISCRETIZATION.decode(labels)
    reconstructed = DEFAULT_DISCRETIZATION.encode(decoded)

    np.testing.assert_array_equal(
        reconstructed,
        labels,
    )


def test_label_boundaries():
    lower = np.array([
        -100.0,
        -50.0,
        -20.0,
        -np.pi / 4,
    ])

    upper = np.array([
        50.0,
        50.0,
        20.0,
        np.pi / 4,
    ])

    assert encode_pose(lower) == 0
    assert encode_pose(upper) == 4799


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        encode_pose([
            80.0,
            0.0,
            0.0,
            0.0,
        ])


def test_preprocessing_shape():
    points = np.zeros((14, 2))
    points[:, 0] = 320.0
    points[:, 1] = 240.0

    mask = np.ones(14, dtype=bool)

    result = preprocess_keypoints(
        points, mask, 640, 480
    )

    assert result.shape == (42,)
    assert result.dtype == np.float32

    np.testing.assert_allclose(
        result.reshape(14, 3)[0],
        [0.5, 0.5, 1.0],
    )


def test_occlusion_preprocessing():
    points = np.full((14, 2), 200.0)
    mask = np.ones(14, dtype=bool)

    points[4] = [np.nan, np.nan]
    mask[4] = False

    result = preprocess_keypoints(
        points, mask, 640, 480
    )

    features = result.reshape(14, 3)

    np.testing.assert_allclose(
        features[4],
        [0.0, 0.0, 0.0],
    )

    assert np.all(np.isfinite(result))


def test_phase1_preprocessing_compatibility():
    scenario = generate_scenario(
        "Approach_L1",
        duration=1.0,
        fps=10,
        seed=42,
    )

    observation = scenario.observations[0]

    a = observation.ann_input()
    b = preprocess_observation(observation)

    np.testing.assert_allclose(a, b, atol=1e-7)


def test_trajectory_split_isolation():
    splits = split_trajectory_ids(
        count=30,
        fractions=(0.7, 0.15, 0.15),
        seed=42,
    )

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)

    assert len(train | val | test) == 30

    assert all(len(ids) > 0 for ids in splits.values())


def test_one_trajectory():
    config = DatasetConfig(
        num_trajectories=6,
        duration_seconds=1.0,
        simulation_fps=10,
    )

    result = generate_one_trajectory(
        trajectory_id=0,
        config=config,
    )

    assert result["X"].shape == (10, 42)
    assert result["y"].shape == (10,)
    assert result["pose_4d"].shape == (10, 4)

    assert validate_features(result["X"])

    assert np.all(result["y"] >= 0)
    assert np.all(result["y"] < 4800)

    reconstructed = DEFAULT_DISCRETIZATION.encode(
        result["pose_4d"]
    )

    np.testing.assert_array_equal(
        reconstructed,
        result["y"],
    )


def test_dataset_reproducibility():
    config = DatasetConfig(
        seed=42,
        num_trajectories=6,
        duration_seconds=1.0,
        simulation_fps=10,
    )

    first, split_a = build_dataset(config)
    second, split_b = build_dataset(config)

    assert split_a == split_b

    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(
            first[name]["X"],
            second[name]["X"],
        )

        np.testing.assert_array_equal(
            first[name]["y"],
            second[name]["y"],
        )


def test_dataset_split_integrity():
    config = DatasetConfig(
        seed=42,
        num_trajectories=9,
        duration_seconds=1.0,
        simulation_fps=10,
    )

    datasets, _ = build_dataset(config)

    id_sets = {
        split: set(np.unique(data["trajectory_id"]))
        for split, data in datasets.items()
    }

    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])

    total = sum(
        len(data["y"])
        for data in datasets.values()
    )

    assert total == 90


def test_save_and_load(tmp_path):
    config = DatasetConfig(
        seed=42,
        num_trajectories=6,
        duration_seconds=1.0,
        simulation_fps=10,
        output_directory=str(tmp_path),
    )

    datasets, split_ids = build_dataset(config)

    manifest = save_dataset(
        config,
        datasets,
        split_ids,
    )

    assert manifest["num_classes"] == 4800
    assert manifest["feature_dimension"] == 42

    for split in ("train", "val", "test"):
        loaded = load_dataset(tmp_path, split)

        np.testing.assert_array_equal(
            loaded["X"],
            datasets[split]["X"],
        )

        np.testing.assert_array_equal(
            loaded["y"],
            datasets[split]["y"],
        )

    with (tmp_path / "manifest.json").open() as file:
        stored = json.load(file)

    assert stored["schema_version"] == "1.0"


def test_invalid_split_configuration():
    with pytest.raises(ValueError):
        DatasetConfig(
            num_trajectories=10,
            train_fraction=0.8,
            val_fraction=0.15,
            test_fraction=0.15,
        )
