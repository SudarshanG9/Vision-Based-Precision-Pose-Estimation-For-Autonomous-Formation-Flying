
"""Command-line benchmark and ablation runner.

Examples:

    python -m scripts.run_benchmarks --mode compare

    python -m scripts.run_benchmarks \
        --mode seeds --seeds 42 43 44

    python -m scripts.run_benchmarks \
        --mode ablation --variable particle_count
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.benchmark import (
    BenchmarkConfig,
    run_benchmark,
)
from evaluation.ablation import (
    ABLATION_GRIDS,
    run_ablation,
    ablation_table,
)
from evaluation.report import (
    aggregate_benchmarks,
    build_markdown_report,
    save_benchmark_outputs,
)
from scripts.run_experiment import make_ann_predictor
from training.train_classifier import (
    load_classifier,
    resolve_device,
)


def load_model_dependencies(
    checkpoint_path,
    calibration_path,
    device_name,
):
    device = resolve_device(device_name)

    model, _ = load_classifier(
        checkpoint_path, device
    )

    with np.load(
        calibration_path,
        allow_pickle=False,
    ) as archive:
        errors = archive["errors"].copy()

    predictor = make_ann_predictor(
        model, device
    )

    return predictor, errors


def make_configuration(args):
    return BenchmarkConfig(
        seed=args.seed,
        duration_seconds=args.duration,
        simulation_fps=args.fps,
        particle_count=args.particles,
        resampling_alpha=args.alpha,
        pixel_noise_std=args.pixel_noise,
        measurement_sigma_px=args.sigma,
        ess_threshold_ratio=args.ess_threshold,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("compare", "seeds", "ablation"),
        default="compare",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pose_classifier_best.pt",
    )

    parser.add_argument(
        "--calibration",
        default="checkpoints/classifier_error_calibration.npz",
    )

    parser.add_argument(
        "--device", default="auto"
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--duration", type=float, default=2.0
    )

    parser.add_argument(
        "--fps", type=int, default=20
    )

    parser.add_argument(
        "--particles", type=int, default=1000
    )

    parser.add_argument(
        "--alpha", type=float, default=0.9
    )

    parser.add_argument(
        "--pixel-noise", type=float, default=1.5
    )

    parser.add_argument(
        "--sigma", type=float, default=2.0
    )

    parser.add_argument(
        "--ess-threshold", type=float, default=0.5
    )

    parser.add_argument(
        "--variable",
        choices=tuple(ABLATION_GRIDS),
        default="particle_count",
    )

    parser.add_argument(
        "--output",
        default="outputs/benchmarks",
    )

    args = parser.parse_args()

    predictor, errors = load_model_dependencies(
        args.checkpoint,
        args.calibration,
        args.device,
    )

    config = make_configuration(args)

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "compare":
        benchmark = run_benchmark(
            config,
            predictor,
            errors,
        )

        paths = save_benchmark_outputs(
            benchmark,
            output_dir,
            f"comparison_seed{args.seed}",
        )

        print("\nBenchmark complete")
        print(json.dumps(paths, indent=2))

    elif args.mode == "seeds":
        benchmarks = []

        for seed in args.seeds:
            seed_config = replace(config, seed=seed)

            benchmark = run_benchmark(
                seed_config,
                predictor,
                errors,
            )

            benchmarks.append(benchmark)

            save_benchmark_outputs(
                benchmark,
                output_dir,
                f"comparison_seed{seed}",
            )

        aggregate = aggregate_benchmarks(
            benchmarks
        )

        aggregate_path = (
            output_dir / "aggregate_results.json"
        )

        aggregate_path.write_text(
            json.dumps(
                aggregate,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        markdown = build_markdown_report(
            benchmarks,
        )

        (output_dir / "aggregate_report.md").write_text(
            markdown,
            encoding="utf-8",
        )

        print(f"Saved aggregate: {aggregate_path}")

    else:
        ablation = run_ablation(
            base_config=config,
            variable=args.variable,
            values=ABLATION_GRIDS[args.variable],
            ann_predictor=predictor,
            calibration_errors=errors,
            estimators=("improved_pf",),
        )

        rows = ablation_table(ablation)

        path = output_dir / (
            f"ablation_{args.variable}.json"
        )

        path.write_text(
            json.dumps(
                {
                    "variable": args.variable,
                    "rows": rows,
                    "experiment": ablation,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print(f"Saved ablation: {path}")


if __name__ == "__main__":
    main()
