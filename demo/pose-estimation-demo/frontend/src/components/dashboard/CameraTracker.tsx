
"use client";

import { Camera } from "lucide-react";

import type { TelemetryFrame } from "@/types/telemetry";

interface Props {
    frame: TelemetryFrame | null;
}

const LANDMARK_NAMES = [
    "Nose",
    "Cockpit",
    "Fuselage Front",
    "Right Root",
    "Right Wingtip",
    "Left Root",
    "Left Wingtip",
    "Fuselage Rear",
    "Right Tailtip",
    "Left Tailtip",
    "Vertical Tail",
    "Tail End",
    "Right Trailing",
    "Left Trailing"
];

const AIRCRAFT_EDGES: [number, number][] = [
    [0, 1], [1, 2], [2, 7], [7, 11],
    [2, 3], [3, 4], [4, 12], [12, 3],
    [2, 5], [5, 6], [6, 13], [13, 5],
    [7, 8], [8, 11],
    [7, 9], [9, 11],
    [7, 10], [10, 11]
];

export default function CameraTracker({ frame }: Props) {
    const points = frame?.observation.keypoints ?? [];
    const visible = frame?.observation.visible ?? [];

    const visibleCount = visible.filter(Boolean).length;

    const gridX = Array.from(
        { length: 9 },
        (_, i) => i * 80
    );

    const gridY = Array.from(
        { length: 7 },
        (_, i) => i * 80
    );

    return (
        <section className="dashboard-panel camera-panel">
            <div className="panel-heading">
                <div>
                    <span className="eyebrow">OPTICAL OBSERVATION</span>
                    <h2>Follower Camera — 2D Feature Tracker</h2>
                </div>
                <Camera size={18} />
            </div>

            <div className="camera-viewport">
                <svg
                    viewBox="0 0 640 480"
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label="Projected aircraft landmarks"
                >
                    <rect
                        x={0}
                        y={0}
                        width={640}
                        height={480}
                        fill="#0b1421"
                    />

                    {gridX.map((x) => (
                        <line
                            key={`x-${x}`}
                            x1={x}
                            y1={0}
                            x2={x}
                            y2={480}
                            stroke="#203349"
                            strokeWidth={1}
                        />
                    ))}

                    {gridY.map((y) => (
                        <line
                            key={`y-${y}`}
                            x1={0}
                            y1={y}
                            x2={640}
                            y2={y}
                            stroke="#203349"
                            strokeWidth={1}
                        />
                    ))}

                    <line
                        x1={320} x2={320}
                        y1={0} y2={480}
                        stroke="#3b596e"
                        strokeDasharray="5 7"
                    />

                    <line
                        x1={0} x2={640}
                        y1={240} y2={240}
                        stroke="#3b596e"
                        strokeDasharray="5 7"
                    />

                    {AIRCRAFT_EDGES.map(([a, b], index) => {
                        const pa = points[a];
                        const pb = points[b];

                        if (!pa || !pb || !visible[a] || !visible[b]) {
                            return null;
                        }

                        return (
                            <line
                                key={index}
                                x1={pa[0]}
                                y1={pa[1]}
                                x2={pb[0]}
                                y2={pb[1]}
                                stroke="#8fa5bf"
                                strokeWidth={2}
                                opacity={0.85}
                            />
                        );
                    })}

                    {points.map((point, index) => {
                        if (!point || !visible[index]) return null;

                        return (
                            <g key={index}>
                                <circle
                                    cx={point[0]}
                                    cy={point[1]}
                                    r={7}
                                    fill="#ff9e61"
                                    opacity={0.17}
                                />
                                <circle
                                    cx={point[0]}
                                    cy={point[1]}
                                    r={3.7}
                                    fill="#ff9e61"
                                    stroke="#ffe0bf"
                                    strokeWidth={0.8}
                                />
                                <text
                                    x={point[0] + 9}
                                    y={point[1] - 8}
                                    fill="#ffbf89"
                                    fontSize={12}
                                    fontWeight={600}
                                >
                                    {index}
                                </text>
                                <title>{LANDMARK_NAMES[index]}</title>
                            </g>
                        );
                    })}

                    {!frame && (
                        <text
                            x={320}
                            y={240}
                            textAnchor="middle"
                            fill="#6f849a"
                            fontSize={17}
                        >
                            Run an experiment to display camera telemetry
                        </text>
                    )}

                    <text
                        x={16}
                        y={460}
                        fill="#9bb3c9"
                        fontSize={13}
                    >
                        640 × 480 | PINHOLE CAMERA
                    </text>
                </svg>
            </div>

            <div className="camera-footer">
                <span>
                    <i className="legend-dot orange" />
                    Visible Landmarks: {visibleCount}/14
                </span>
                <span>
                    Missing/Occluded: {14 - visibleCount}
                </span>
                <span>
                    Frame {frame?.frame_index ?? 0}
                </span>
            </div>
        </section>
    );
}
