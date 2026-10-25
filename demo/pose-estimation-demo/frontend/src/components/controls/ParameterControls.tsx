
"use client";

import type {
    ExperimentParameters
} from "@/types/telemetry";

interface Props {
    parameters: ExperimentParameters;
    disabled?: boolean;
    onChange: <K extends keyof ExperimentParameters>(
        key: K,
        value: ExperimentParameters[K]
    ) => void;
}

interface SliderDefinition {
    key: keyof ExperimentParameters;
    label: string;
    min: number;
    max: number;
    step: number;
    unit?: string;
}

const SLIDERS: SliderDefinition[] = [
    {
        key: "particle_count",
        label: "Particle Count",
        min: 100,
        max: 5000,
        step: 100
    },
    {
        key: "resampling_alpha",
        label: "Resampling Alpha",
        min: 0,
        max: 1,
        step: 0.05
    },
    {
        key: "pixel_noise_std",
        label: "Pixel Noise",
        min: 0,
        max: 8,
        step: 0.25,
        unit: "px"
    },
    {
        key: "position_noise_std",
        label: "Dynamics Noise",
        min: 0,
        max: 3,
        step: 0.1,
        unit: "m"
    },
    {
        key: "measurement_sigma_px",
        label: "Measurement Sigma",
        min: 0.5,
        max: 8,
        step: 0.25,
        unit: "px"
    },
    {
        key: "ess_threshold_ratio",
        label: "ESS Threshold",
        min: 0.1,
        max: 1,
        step: 0.05
    }
];

export default function ParameterControls({
    parameters,
    disabled = false,
    onChange
}: Props) {
    return (
        <div className="parameters">
            {SLIDERS.map((slider) => {
                const value = parameters[slider.key];

                return (
                    <label className="parameter" key={slider.key}>
                        <span className="parameter-heading">
                            <span>{slider.label}</span>
                            <strong>
                                {value}
                                {slider.unit ? ` ${slider.unit}` : ""}
                            </strong>
                        </span>

                        <input
                            type="range"
                            min={slider.min}
                            max={slider.max}
                            step={slider.step}
                            value={value}
                            disabled={disabled}
                            onChange={(event) =>
                                onChange(
                                    slider.key,
                                    Number(event.target.value)
                                )
                            }
                        />
                    </label>
                );
            })}

            <div className="parameter-grid">
                <label>
                    <span>Duration (s)</span>
                    <input
                        type="number"
                        min={0.5}
                        max={60}
                        step={0.5}
                        value={parameters.duration_seconds}
                        disabled={disabled}
                        onChange={(event) =>
                            onChange(
                                "duration_seconds",
                                Number(event.target.value)
                            )
                        }
                    />
                </label>

                <label>
                    <span>Simulation FPS</span>
                    <input
                        type="number"
                        min={2}
                        max={200}
                        value={parameters.simulation_fps}
                        disabled={disabled}
                        onChange={(event) =>
                            onChange(
                                "simulation_fps",
                                Number(event.target.value)
                            )
                        }
                    />
                </label>

                <label>
                    <span>Random Seed</span>
                    <input
                        type="number"
                        min={0}
                        step={1}
                        value={parameters.seed}
                        disabled={disabled}
                        onChange={(event) =>
                            onChange(
                                "seed",
                                Number(event.target.value)
                            )
                        }
                    />
                </label>
            </div>
        </div>
    );
}
