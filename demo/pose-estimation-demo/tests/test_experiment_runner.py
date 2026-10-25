
"""Consolidated Phase 6 evaluation tests.

Run:
    python -m pytest tests/test_experiment_runner.py -v
"""

import json

import numpy as np
import pytest

from dataclasses import replace

from estimators.interfaces import Pose3D
from evaluation.metrics import (
    position_error,
    attitude_error,
    summarize_values,
    effective_sample_size,
    summarize_frames,
    paired_position_difference,
)
from evaluation.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    ESTIMATOR_NAMES,
)
from evaluation.ablation import (
    run_ablation,
    ablation_table,
    available_ablations,
)
from evaluation.report import (
    aggregate_benchmarks,
    compare_estimators,
    build_markdown_report,
    save_benchmark_outputs,
)
from models.labels import encode_pose
from simulation.transforms import euler_to_quaternion


@pytest.fixture
def phase6_truth():
    return Pose3D(
        position=[30.0, 2.0, -1.0],
        quaternion=euler_to_quaternion(
            roll=0.05,
            pitch=0.0,
            yaw=0.0,
        ),
        frame_from="L",
        frame_to="F",
    )


@pytest.fixture
def phase6_predictor():
    """
    Deterministic constant predictor for integration testing.

    This is NOT the real trained ANN.
    Never use it for reported scientific results.
    """
    label = encode_pose([
        35.0,
        2.0,
        -1.0,
        0.0,
    ])

    return lambda observation: label


@pytest.fixture
def phase6_calibration():
    rng = np.random.default_rng(42)

    return np.column_stack([
        rng.normal(0, 3.0, 100),
        rng.normal(0, 1.0, 100),
        rng.normal(0, 1.0, 100),
        rng.normal(0, 0.03, 100),
    ])


@pytest.fixture
def phase6_config():
    return BenchmarkConfig(
        seed=42,
        duration_seconds=0.5,
        simulation_fps=10,
        particle_count=64,
        pixel_noise_std=0.5,
        measurement_sigma_px=2.0,
    )


def test_pose_error_metrics(phase6_truth):
    truth = phase6_truth

    estimate = Pose3D(
        position=truth.position + [3.0, 4.0, 0.0],
        quaternion=truth.quaternion.copy(),
        frame_from="L",
        frame_to="F",
    )

    assert position_error(truth, estimate) == pytest.approx(5.0)
    assert attitude_error(truth, estimate) == pytest.approx(0.0)


def test_metric_statistics():
    statistics = summarize_values(
        [1, 2, 3, 4, 5]
    )

    assert statistics["count"] == 5
    assert statistics["mean"] == pytest.approx(3.0)
    assert statistics["median"] == pytest.approx(3.0)
    assert statistics["max"] == pytest.approx(5.0)


def test_effective_sample_size():
    uniform = np.ones(100) / 100

    assert effective_sample_size(
        uniform
    ) == pytest.approx(100.0)

    concentrated = np.zeros(100)
    concentrated[0] = 1.0

    assert effective_sample_size(
        concentrated
    ) == pytest.approx(1.0)


def test_failure_accounting():
    frames = [
        {
            "success": True,
            "position_error_m": 2.0,
            "attitude_error_deg": 1.0,
            "runtime_ms": 5.0,
            "reprojection_error_px": 2.0,
        },
        {
            "success": False,
            "position_error_m": None,
            "attitude_error_deg": None,
            "runtime_ms": None,
            "reprojection_error_px": None,
        },
    ]

    summary = summarize_frames(frames)

    assert summary["frame_count"] == 2
    assert summary["failed_frames"] == 1
    assert summary["failure_rate"] == pytest.approx(0.5)
    assert summary["position_error_m"]["mean"] == 2.0


def test_paired_error_comparison():
    first = [
        {"frame_index": 0, "position_error_m": 5.0},
        {"frame_index": 1, "position_error_m": 3.0},
    ]

    second = [
        {"frame_index": 0, "position_error_m": 2.0},
        {"frame_index": 1, "position_error_m": 1.0},
    ]

    difference = paired_position_difference(
        first, second
    )

    assert difference["count"] == 2
    assert difference["mean"] == pytest.approx(2.5)


def test_benchmark_configuration(phase6_config):
    assert phase6_config.particle_count == 64

    with pytest.raises(ValueError):
        BenchmarkConfig(particle_count=0)

    assert set(ESTIMATOR_NAMES) == {
        "paper_pf",
        "improved_pf",
        "geometric",
        "hybrid",
    }


def test_complete_benchmark(
    phase6_config,
    phase6_predictor,
    phase6_calibration,
):
    result = run_benchmark(
        config=phase6_config,
        ann_predictor=phase6_predictor,
        calibration_errors=phase6_calibration,
        estimators=("paper_pf", "improved_pf", "geometric"),
    )

    assert result["metadata"]["frame_count"] == 5

    assert len(result["ground_truth"]) == 5
    assert len(result["observations"]) == 5

    for name in ("paper_pf", "improved_pf", "geometric"):
        experiment = result["results"][name]

        assert len(experiment["frames"]) == 5

        summary = experiment["summary"]

        assert summary["frame_count"] == 5
        assert 0 <= summary["failure_rate"] <= 1

    # The observation stream belongs to the common benchmark,
    # rather than being regenerated independently per method.
    assert len(result["observations"]) == 5


def test_benchmark_reproducibility(
    phase6_config,
    phase6_predictor,
    phase6_calibration,
):
    results = []

    for _ in range(2):
        result = run_benchmark(
            config=phase6_config,
            ann_predictor=phase6_predictor,
            calibration_errors=phase6_calibration,
            estimators=("paper_pf",),
        )

        positions = [
            frame["estimate"]["position"]
            for frame in result["results"]["paper_pf"]["frames"]
        ]

        results.append(np.asarray(positions))

    np.testing.assert_allclose(
        results[0],
        results[1],
        atol=1e-12,
        rtol=0,
    )


def test_ablation_execution(
    phase6_config,
    phase6_predictor,
    phase6_calibration,
):
    result = run_ablation(
        base_config=phase6_config,
        variable="particle_count",
        values=[32, 64],
        ann_predictor=phase6_predictor,
        calibration_errors=phase6_calibration,
        estimators=("improved_pf",),
    )

    assert len(result["experiments"]) == 2

    table = ablation_table(result)

    assert len(table) == 2

    assert table[0]["value"] == 32
    assert table[1]["value"] == 64

    assert "particle_count" in available_ablations()


def test_report_generation(
    phase6_config,
    phase6_predictor,
    phase6_calibration,
):
    result = run_benchmark(
        phase6_config,
        phase6_predictor,
        phase6_calibration,
        estimators=("paper_pf", "improved_pf"),
    )

    aggregate = aggregate_benchmarks([result])

    assert aggregate["experiment_count"] == 1

    assert "paper_pf" in aggregate["estimators"]

    comparison = compare_estimators(result)

    assert "improved_pf" in comparison

    markdown = build_markdown_report([result])

    assert "Aggregate Results" in markdown
    assert "paper_pf" in markdown
    assert "improved_pf" in markdown


def test_benchmark_file_output(
    tmp_path,
    phase6_config,
    phase6_predictor,
    phase6_calibration,
):
    result = run_benchmark(
        phase6_config,
        phase6_predictor,
        phase6_calibration,
        estimators=("paper_pf",),
    )

    paths = save_benchmark_outputs(
        result,
        tmp_path,
        "phase6_test",
    )

    with open(paths["json"], encoding="utf-8") as handle:
        stored = json.load(handle)

    assert stored["metadata"]["seed"] == 42
    assert "paper_pf" in stored["results"]

    with open(paths["markdown"], encoding="utf-8") as handle:
        markdown = handle.read()

    assert "Pose Estimation Experiment" in markdown
