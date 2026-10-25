
"""Scientific benchmark aggregation and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    summarize_values,
    paired_position_difference,
)


def aggregate_benchmarks(benchmarks):
    """Aggregate estimator metrics across independent experiments."""
    if not benchmarks:
        raise ValueError("No benchmarks supplied")

    estimator_names = sorted({
        name
        for benchmark in benchmarks
        for name in benchmark["results"]
    })

    aggregated = {}

    for name in estimator_names:
        trials = [
            benchmark["results"][name]["summary"]
            for benchmark in benchmarks
            if name in benchmark["results"]
        ]

        def metric(field, statistic="mean"):
            return [
                trial[field][statistic]
                for trial in trials
                if trial[field][statistic] is not None
            ]

        aggregated[name] = {
            "trial_count": len(trials),
            "position_error_m": summarize_values(
                metric("position_error_m")
            ),
            "attitude_error_deg": summarize_values(
                metric("attitude_error_deg")
            ),
            "reprojection_error_px": summarize_values(
                metric("reprojection_error_px")
            ),
            "attempted_runtime_ms": summarize_values(
                metric("attempted_runtime_ms")
            ),
            "failure_rate": summarize_values([
                trial["failure_rate"]
                for trial in trials
            ]),
        }

    return {
        "experiment_count": len(benchmarks),
        "estimators": aggregated,
    }


def compare_estimators(
    benchmark,
    baseline="paper_pf",
):
    """Produce within-experiment paired position-error comparisons."""

    results = benchmark["results"]

    if baseline not in results:
        raise ValueError("Baseline not found in benchmark")

    comparison = {}

    baseline_frames = results[baseline]["frames"]

    for name, result in results.items():
        if name == baseline:
            continue

        comparison[name] = paired_position_difference(
            baseline_frames,
            result["frames"],
        )

    return comparison


def format_value(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_markdown_report(
    benchmarks,
    title="Vision-Based Pose Estimation Benchmark",
):
    """Create a structured Markdown report from actual measurements."""

    aggregate = aggregate_benchmarks(benchmarks)

    lines = [
        f"# {title}",
        "",
        "## Experimental Scope",
        "",
        f"Independent benchmark runs: {len(benchmarks)}",
        "",
        "Data source: synthetic aircraft observations.",
        "",
        "Ground-truth relative poses are used for evaluation only.",
        "",
        "The ANN was trained on the Phase 2 synthetic dataset.",
        "",
        "## Aggregate Results",
        "",
        "| Estimator | Position Error (m) | "
        "Attitude Error (deg) | Failure Rate | "
        "Runtime (ms/frame) |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, result in aggregate["estimators"].items():
        position = result["position_error_m"]["mean"]
        attitude = result["attitude_error_deg"]["mean"]
        failure = result["failure_rate"]["mean"]
        runtime = result["attempted_runtime_ms"]["mean"]

        lines.append(
            f"| {name} | "
            f"{format_value(position)} | "
            f"{format_value(attitude)} | "
            f"{format_value(failure)} | "
            f"{format_value(runtime)} |"
        )

    lines.extend([
        "",
        "## Evaluation Methodology",
        "",
        "- Estimators receive identical observations within each run.",
        "- Ground truth is not supplied to estimation methods.",
        "- Evaluation includes position, attitude, and reprojection error.",
        "- Failure rates are reported separately from successful-frame errors.",
        "- Runtime includes estimator processing, not browser rendering.",
        "- All numerical results originate from recorded experiments.",
        "",
        "## Limitations",
        "",
        "- Aircraft geometry and imagery are synthetic.",
        "- The evaluation trajectories cover a restricted pose distribution.",
        "- The original paper's full dataset is not available.",
        "- The PF dynamics contain documented reproduction assumptions.",
        "- Improved ANN reseeding is not fully importance-corrected.",
        "- Results should not be generalized to flight hardware.",
        "",
        "## Interpretation",
        "",
        "Algorithmic improvements must be supported by paired experiments.",
        "Lower error in one synthetic run is not proof of universal superiority.",
        "",
    ])

    return "\n".join(lines)


def save_benchmark_outputs(
    benchmark,
    output_directory,
    experiment_id,
):
    """Save benchmark JSON and per-run Markdown report."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / f"{experiment_id}.json"
    markdown_path = directory / f"{experiment_id}.md"

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            benchmark,
            file,
            indent=2,
            allow_nan=False,
        )

    report = build_markdown_report(
        [benchmark],
        title=f"Pose Estimation Experiment: {experiment_id}",
    )

    markdown_path.write_text(
        report,
        encoding="utf-8",
    )

    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
