
"""Controlled ablation studies for pose estimation."""

from __future__ import annotations

from dataclasses import asdict, replace

from evaluation.benchmark import (
    BenchmarkConfig,
    run_benchmark,
)


ABLATION_GRIDS = {
    "particle_count": [250, 500, 1000, 5000],
    "resampling_alpha": [0.5, 0.7, 0.9, 1.0],
    "pixel_noise_std": [0.5, 1.5, 3.0, 5.0],
    "measurement_sigma_px": [1.0, 2.0, 4.0, 8.0],
    "ess_threshold_ratio": [0.25, 0.5, 0.75],
}


def run_ablation(
    base_config: BenchmarkConfig,
    variable: str,
    values,
    ann_predictor,
    calibration_errors,
    estimators=("improved_pf",),
):
    """Change exactly one benchmark parameter per run."""

    if not hasattr(base_config, variable):
        raise ValueError(
            f"Unknown benchmark parameter: {variable}"
        )

    if not values:
        raise ValueError(
            "Ablation requires at least one parameter value"
        )

    experiments = []

    for value in values:
        config = replace(
            base_config,
            **{variable: value},
        )

        result = run_benchmark(
            config=config,
            ann_predictor=ann_predictor,
            calibration_errors=calibration_errors,
            estimators=estimators,
        )

        experiments.append({
            "parameter_value": value,
            "configuration": asdict(config),
            "results": result,
        })

    return {
        "ablation_variable": variable,
        "base_configuration": asdict(base_config),
        "estimator_names": list(estimators),
        "experiments": experiments,
    }


def ablation_table(ablation_result):
    """Create a flat numerical summary for reporting."""

    rows = []

    variable = ablation_result["ablation_variable"]

    for experiment in ablation_result["experiments"]:
        value = experiment["parameter_value"]

        for name, result in (
            experiment["results"]["results"].items()
        ):
            summary = result["summary"]

            rows.append({
                "variable": variable,
                "value": value,
                "estimator": name,
                "position_error_mean_m": summary[
                    "position_error_m"
                ]["mean"],
                "attitude_error_mean_deg": summary[
                    "attitude_error_deg"
                ]["mean"],
                "failure_rate": summary["failure_rate"],
                "runtime_mean_ms": summary[
                    "attempted_runtime_ms"
                ]["mean"],
            })

    return rows


def available_ablations():
    return {
        key: list(values)
        for key, values in ABLATION_GRIDS.items()
    }
