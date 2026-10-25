
"use client";

import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { usePlayback } from "@/hooks/usePlayback";
import { useExperimentStore } from "@/store/experimentStore";

import StatusBar from "./StatusBar";
import ControlPanel from "./ControlPanel";
import CameraTracker from "./CameraTracker";
import Trajectory3D from "./Trajectory3D";
import ErrorTelemetry from "./ErrorTelemetry";

import type {
    ExperimentRequest,
    ScenarioInfo
} from "@/types/telemetry";

export default function DashboardLayout() {
    const [scenarios, setScenarios] = useState<ScenarioInfo[]>([
        {
            id: "distribution_compatible_approach",
            name: "Nominal Formation Approach",
            description: "Synthetic research benchmark",
            duration_seconds: 10,
            simulation_fps: 50
        }
    ]);

    const [healthStatus, setHealthStatus] = useState<
        "online" | "offline" | "checking"
    >("checking");

    const activeRequest = useRef(0);

    const {
        scenarioId,
        estimators,
        parameters,
        telemetry,
        summary,
        progress,
        error,
        begin,
        setExperimentId,
        setProgress,
        setResults,
        setError
    } = useExperimentStore();

    const playback = usePlayback(
        telemetry?.frame_count ?? 0,
        telemetry?.simulation_fps ?? 20,
        telemetry?.playback_fps ?? 30
    );

    const frame =
        telemetry?.frames[playback.frameIndex] ?? null;

    const status = progress?.status ?? "idle";

    useEffect(() => {
        let cancelled = false;

        Promise.all([
            api.health(),
            api.scenarios()
        ])
            .then(([, availableScenarios]) => {
                if (cancelled) return;

                setScenarios(availableScenarios);
                setHealthStatus("online");
            })
            .catch(() => {
                if (!cancelled) {
                    setHealthStatus("offline");
                }
            });

        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        return () => {
            activeRequest.current += 1;
        };
    }, []);

    async function runExperiment() {
        if (estimators.length === 0) {
            setError("Select at least one estimator.");
            return;
        }

        const requestVersion = ++activeRequest.current;

        begin();

        const payload: ExperimentRequest = {
            scenario_id: scenarioId,
            estimators: [...estimators],
            parameters: { ...parameters }
        };

        try {
            const created = await api.createExperiment(payload);

            if (activeRequest.current !== requestVersion) return;

            setExperimentId(created.experiment_id);

            const maximumPolls = 600;

            for (let poll = 0; poll < maximumPolls; poll++) {
                if (activeRequest.current !== requestVersion) return;

                const current = await api.experimentStatus(
                    created.experiment_id
                );

                if (activeRequest.current !== requestVersion) return;

                setProgress(current);

                if (current.status === "completed") {
                    const [resultTelemetry, resultSummary] =
                        await Promise.all([
                            api.telemetry(created.experiment_id),
                            api.summary(created.experiment_id)
                        ]);

                    if (activeRequest.current !== requestVersion) return;

                    setResults(resultTelemetry, resultSummary);
                    return;
                }

                if (current.status === "failed") {
                    setError(
                        current.error ?? "Experiment execution failed."
                    );
                    return;
                }

                await new Promise<void>((resolve) => {
                    window.setTimeout(resolve, 1000);
                });
            }

            setError(
                "Experiment status polling timed out. " +
                "The computation may still be running on the backend."
            );
        } catch (exception) {
            if (activeRequest.current !== requestVersion) return;

            setError(
                exception instanceof Error
                    ? exception.message
                    : "Unexpected API error"
            );
        }
    }

    return (
        <div className="dashboard-shell">
            <StatusBar
                status={status}
                frame={frame}
                frameCount={telemetry?.frame_count ?? 0}
                simulationFps={telemetry?.simulation_fps ?? null}
                particleCount={parameters.particle_count}
            />

            <main className="dashboard-main">
                <ControlPanel
                    scenarios={scenarios}
                    status={status}
                    frame={frame}
                    frameIndex={playback.frameIndex}
                    frameCount={telemetry?.frame_count ?? 0}
                    playing={playback.playing}
                    playbackFps={playback.playbackFps}
                    onTogglePlaying={playback.togglePlaying}
                    onStop={playback.stop}
                    onSeek={playback.seek}
                    onPlaybackFpsChange={playback.setPlaybackFps}
                    onRun={runExperiment}
                />

                <div className="dashboard-content">
                    <div className="visualization-grid">
                        <CameraTracker frame={frame} />

                        <Trajectory3D
                            telemetry={telemetry}
                            frameIndex={playback.frameIndex}
                        />
                    </div>

                    <ErrorTelemetry
                        telemetry={telemetry}
                        frameIndex={playback.frameIndex}
                    />

                    <footer className="dashboard-footer">
                        <span>
                            Backend:{" "}
                            <strong className={`backend-${healthStatus}`}>
                                {healthStatus}
                            </strong>
                        </span>

                        <span>
                            Experiment:{" "}
                            {telemetry?.experiment_id
                                ? telemetry.experiment_id.slice(0, 8)
                                : "Not started"}
                        </span>

                        <span>
                            Simulation:{" "}
                            {telemetry
                                ? `${telemetry.frame_count} recorded frames`
                                : "No telemetry"}
                        </span>

                        <span>
                            Mean Improved PF Error:{" "}
                            {summary?.metrics.improved_pf
                                ?.position_error_m.mean?.toFixed(2) ?? "—"}{" "}
                            m
                        </span>
                    </footer>

                    {error && (
                        <div className="dashboard-error" role="alert">
                            {error}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
