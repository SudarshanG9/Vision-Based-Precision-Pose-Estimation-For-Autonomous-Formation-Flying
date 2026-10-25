
"use client";

import {
    Activity,
    CircleCheck,
    CircleDashed,
    CircleX
} from "lucide-react";

import type {
    ExperimentStatus,
    TelemetryFrame
} from "@/types/telemetry";

interface Props {
    status: ExperimentStatus | "idle";
    frame: TelemetryFrame | null;
    frameCount: number;
    simulationFps: number | null;
    particleCount: number;
}

export default function StatusBar({
    status,
    frame,
    frameCount,
    simulationFps,
    particleCount
}: Props) {
    const stateIcon =
        status === "completed" ? (
            <CircleCheck size={15} />
        ) : status === "failed" ? (
            <CircleX size={15} />
        ) : (
            <CircleDashed size={15} />
        );

    return (
        <header className="status-bar">
            <div className="brand">
                <div className="brand-icon">
                    <Activity size={22} />
                </div>
                <div>
                    <h1>
                        Autonomous Formation Flight
                    </h1>
                    <p>
                        Pose Estimation &amp; Telemetry Dashboard
                    </p>
                </div>
            </div>

            <div className="status-metrics">
                <div className="status-metric">
                    <span>System</span>
                    <strong className={`state-${status}`}>
                        {stateIcon}
                        {status.toUpperCase()}
                    </strong>
                </div>

                <div className="status-metric">
                    <span>Time</span>
                    <strong>
                        {frame
                            ? `${frame.timestamp.toFixed(2)} s`
                            : "—"}
                    </strong>
                </div>

                <div className="status-metric">
                    <span>Frame</span>
                    <strong>
                        {frame
                            ? `${frame.frame_index + 1}/${frameCount}`
                            : "0/0"}
                    </strong>
                </div>

                <div className="status-metric">
                    <span>Simulation</span>
                    <strong>
                        {simulationFps ?? "—"} Hz
                    </strong>
                </div>

                <div className="status-metric">
                    <span>Particles</span>
                    <strong>{particleCount.toLocaleString()}</strong>
                </div>
            </div>
        </header>
    );
}
