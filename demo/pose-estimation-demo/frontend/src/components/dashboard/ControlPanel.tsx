
"use client";

import { FlaskConical, PlayCircle } from "lucide-react";

import ParameterControls from "@/components/controls/ParameterControls";
import PlaybackControls from "@/components/controls/PlaybackControls";

import { useExperimentStore } from "@/store/experimentStore";

import {
    ESTIMATOR_COLORS,
    ESTIMATOR_NAMES
} from "@/types/telemetry";

import type {
    EstimatorId,
    ExperimentStatus,
    ScenarioInfo,
    TelemetryFrame
} from "@/types/telemetry";

interface Props {
    scenarios: ScenarioInfo[];
    status: ExperimentStatus | "idle";
    frame: TelemetryFrame | null;
    frameIndex: number;
    frameCount: number;
    playing: boolean;
    playbackFps: number;
    onTogglePlaying: () => void;
    onStop: () => void;
    onSeek: (frame: number) => void;
    onPlaybackFpsChange: (fps: number) => void;
    onRun: () => void;
}

const ESTIMATORS: EstimatorId[] = [
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid"
];

function formatted(value: number | null | undefined, unit = "") {
    return value === null || value === undefined
        ? "—"
        : `${value.toFixed(2)}${unit}`;
}

export default function ControlPanel({
    scenarios,
    status,
    frame,
    frameIndex,
    frameCount,
    playing,
    playbackFps,
    onTogglePlaying,
    onStop,
    onSeek,
    onPlaybackFpsChange,
    onRun
}: Props) {
    const {
        scenarioId,
        estimators,
        parameters,
        isSubmitting,
        error,
        progress,
        setScenario,
        toggleEstimator,
        setParameter
    } = useExperimentStore();

    const running =
        status === "running" ||
        status === "pending" ||
        isSubmitting;

    return (
        <aside className="control-panel">
            <div className="panel-heading">
                <div>
                    <span className="eyebrow">EXPERIMENT CONTROLS</span>
                    <h2>Flight Configuration</h2>
                </div>
                <FlaskConical size={19} />
            </div>

            <section className="control-section">
                <h3>Scenario</h3>

                <select
                    value={scenarioId}
                    disabled={running}
                    onChange={(event) =>
                        setScenario(event.target.value)
                    }
                >
                    {scenarios.map((scenario) => (
                        <option value={scenario.id} key={scenario.id}>
                            {scenario.name}
                        </option>
                    ))}
                </select>
            </section>

            <section className="control-section">
                <h3>Estimators</h3>

                <div className="estimator-options">
                    {ESTIMATORS.map((id) => (
                        <label className="estimator-option" key={id}>
                            <input
                                type="checkbox"
                                checked={estimators.includes(id)}
                                disabled={running}
                                onChange={() => toggleEstimator(id)}
                            />
                            <span
                                className="estimator-dot"
                                style={{
                                    background: ESTIMATOR_COLORS[id]
                                }}
                            />
                            <span>{ESTIMATOR_NAMES[id]}</span>
                        </label>
                    ))}
                </div>
            </section>

            <section className="control-section">
                <PlaybackControls
                    frameIndex={frameIndex}
                    frameCount={frameCount}
                    playing={playing}
                    playbackFps={playbackFps}
                    onToggle={onTogglePlaying}
                    onStop={onStop}
                    onSeek={onSeek}
                    onPlaybackFpsChange={onPlaybackFpsChange}
                />
            </section>

            <section className="control-section">
                <h3>Filter Parameters</h3>
                <ParameterControls
                    parameters={parameters}
                    disabled={running}
                    onChange={setParameter}
                />
            </section>

            <button
                type="button"
                className="run-experiment"
                disabled={running || estimators.length === 0}
                onClick={onRun}
            >
                <PlayCircle size={17} />
                {running ? "Running Experiment..." : "Run Experiment"}
            </button>

            {running && (
                <div className="progress-track">
                    <div
                        style={{
                            width: `${(progress?.progress ?? 0) * 100}%`
                        }}
                    />
                </div>
            )}

            {error && (
                <p className="control-error" role="alert">
                    {error}
                </p>
            )}

            <section className="control-section live-readout">
                <h3>Live Estimation</h3>

                <div className="readout-line">
                    <span>ANN Class</span>
                    <strong>{frame?.ann?.class_id ?? "—"}</strong>
                </div>

                <div className="readout-line">
                    <span>ANN Confidence</span>
                    <strong>
                        {frame?.ann?.confidence === null ||
                            frame?.ann?.confidence === undefined
                            ? "—"
                            : `${(frame.ann.confidence * 100).toFixed(1)}%`}
                    </strong>
                </div>

                <div className="readout-line">
                    <span>Original PF Error</span>
                    <strong>
                        {formatted(
                            frame?.errors.paper_pf?.position_error_m,
                            " m"
                        )}
                    </strong>
                </div>

                <div className="readout-line">
                    <span>Improved PF Error</span>
                    <strong>
                        {formatted(
                            frame?.errors.improved_pf?.position_error_m,
                            " m"
                        )}
                    </strong>
                </div>
            </section>
        </aside>
    );
}
