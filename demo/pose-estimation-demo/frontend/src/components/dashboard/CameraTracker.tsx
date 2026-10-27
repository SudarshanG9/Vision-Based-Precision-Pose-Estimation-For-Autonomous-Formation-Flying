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

/**
 * Existing geometry meaning:
 * 0 Nose
 * 1 Cockpit
 * 2 Fuselage Front
 * 3 Right Root
 * 4 Right Wingtip
 * 5 Left Root
 * 6 Left Wingtip
 * 7 Fuselage Rear
 * 8 Right Tailtip
 * 9 Left Tailtip
 * 10 Vertical Tail
 * 11 Tail End
 * 12 Right Trailing
 * 13 Left Trailing
 */

const AIRCRAFT_EDGES: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 7],
    [7, 11],

    [2, 3],
    [3, 4],
    [4, 12],
    [12, 3],

    [2, 5],
    [5, 6],
    [6, 13],
    [13, 5],

    [7, 8],
    [8, 11],

    [7, 9],
    [9, 11],

    [7, 10],
    [10, 11]
];

const LABEL_OFFSETS: [number, number][] = [
    [8, -10],   // 0
    [8, -10],   // 1
    [8, 12],    // 2
    [10, -10],  // 3
    [10, -12],  // 4
    [-18, -10], // 5
    [-18, -10], // 6
    [8, -12],   // 7
    [10, -10],  // 8
    [-18, -10], // 9
    [10, -10],  // 10
    [10, -10],  // 11
    [10, -10],  // 12
    [-18, -10]  // 13
];

function isVisible(
    points: (readonly [number, number] | null)[],
    visible: boolean[],
    index: number
) {
    return Boolean(visible[index] && points[index]);
}

function polygonPoints(
    indices: number[],
    points: (readonly [number, number] | null)[],
    visible: boolean[]
) {
    const valid = indices.every((idx) => isVisible(points, visible, idx));
    if (!valid) return null;

    return indices
        .map((idx) => {
            const p = points[idx]!;
            return `${p[0]},${p[1]}`;
        })
        .join(" ");
}

function pathFromIndices(
    indices: number[],
    points: (readonly [number, number] | null)[],
    visible: boolean[]
) {
    const valid = indices.every((idx) => isVisible(points, visible, idx));
    if (!valid) return null;

    const coords = indices.map((idx) => points[idx]!);

    return coords
        .map((p, i) =>
            `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`
        )
        .join(" ");
}

export default function CameraTracker({ frame }: Props) {
    const points = frame?.observation.keypoints ?? [];
    const visible = frame?.observation.visible ?? [];

    const visibleCount = visible.filter(Boolean).length;

    const gridX = Array.from({ length: 9 }, (_, i) => i * 80);
    const gridY = Array.from({ length: 7 }, (_, i) => i * 80);

    const fuselageSpine = pathFromIndices(
        [0, 1, 2, 7, 11],
        points,
        visible
    );

    const rightWing = polygonPoints(
        [2, 3, 4, 12],
        points,
        visible
    );

    const leftWing = polygonPoints(
        [2, 5, 6, 13],
        points,
        visible
    );

    const rightTail = polygonPoints(
        [7, 8, 11],
        points,
        visible
    );

    const leftTail = polygonPoints(
        [7, 9, 11],
        points,
        visible
    );

    const verticalTail = polygonPoints(
        [7, 10, 11],
        points,
        visible
    );

    // A slightly wider "body ribbon" makes the fuselage easier to perceive.
    // It uses wing-root points plus nose/tail line.
    const bodyShell = polygonPoints(
        [0, 2, 3, 7, 5],
        points,
        visible
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
                    <defs>
                        <filter id="cameraGlow">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                            <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>

                        <radialGradient id="landmarkHalo" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="#ffb17a" stopOpacity="0.85" />
                            <stop offset="100%" stopColor="#ffb17a" stopOpacity="0" />
                        </radialGradient>
                    </defs>

                    <rect
                        x={0}
                        y={0}
                        width={640}
                        height={480}
                        fill="#07111d"
                    />

                    {/* ── Top-down fighter jet silhouette (background layer) ── */}
                    {/*
                     *  Pure SVG paths centred on (320, 240).
                     *  The jet is oriented nose-up (towards y=0) to match the
                     *  camera frame convention used by the project.
                     *  All shapes are drawn at very low opacity so they never
                     *  interfere with the feature-point overlay above.
                     */}
                    <g
                        transform="translate(320, 245)"
                        opacity={0.22}
                        style={{ pointerEvents: "none" }}
                    >
                        {/* Glow filter already defined in <defs> above */}

                        {/* ── Fuselage body ── */}
                        {/* Main body: narrow nose → wide mid-section → tapered tail */}
                        <path
                            d="
                              M 0,-210
                              C 6,-195 14,-170 16,-140
                              L 22,-80
                              C 28,-50 32,-20 30,20
                              L 28,70
                              C 26,100 20,130 16,160
                              L 10,200
                              L 0,215
                              L -10,200
                              L -16,160
                              C -20,130 -26,100 -28,70
                              L -30,20
                              C -32,-20 -28,-50 -22,-80
                              L -16,-140
                              C -14,-170 -6,-195 0,-210
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.07}
                            stroke="#00e5ff"
                            strokeOpacity={0.6}
                            strokeWidth={1.2}
                            filter="url(#cameraGlow)"
                        />

                        {/* Cockpit canopy */}
                        <ellipse
                            cx={0}
                            cy={-155}
                            rx={9}
                            ry={22}
                            fill="#00e5ff"
                            fillOpacity={0.12}
                            stroke="#00e5ff"
                            strokeOpacity={0.5}
                            strokeWidth={0.9}
                        />

                        {/* ── Right delta wing ── */}
                        <path
                            d="
                              M 16,-60
                              L 200,-10
                              L 220,55
                              L 185,70
                              L 28,70
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.55}
                            strokeWidth={1.1}
                        />

                        {/* Right wing leading-edge detail line */}
                        <line
                            x1={16} y1={-60}
                            x2={200} y2={-10}
                            stroke="#00e5ff"
                            strokeOpacity={0.4}
                            strokeWidth={0.7}
                        />

                        {/* ── Left delta wing ── */}
                        <path
                            d="
                              M -16,-60
                              L -200,-10
                              L -220,55
                              L -185,70
                              L -28,70
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.55}
                            strokeWidth={1.1}
                        />

                        {/* Left wing leading-edge detail line */}
                        <line
                            x1={-16} y1={-60}
                            x2={-200} y2={-10}
                            stroke="#00e5ff"
                            strokeOpacity={0.4}
                            strokeWidth={0.7}
                        />

                        {/* ── Right canard (forward small fin) ── */}
                        <path
                            d="
                              M 16,-100
                              L 85,-80
                              L 70,-55
                              L 22,-65
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.45}
                            strokeWidth={0.9}
                        />

                        {/* ── Left canard ── */}
                        <path
                            d="
                              M -16,-100
                              L -85,-80
                              L -70,-55
                              L -22,-65
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.45}
                            strokeWidth={0.9}
                        />

                        {/* ── Right horizontal tail stabiliser ── */}
                        <path
                            d="
                              M 20,140
                              L 110,115
                              L 115,160
                              L 28,175
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.5}
                            strokeWidth={0.9}
                        />

                        {/* ── Left horizontal tail stabiliser ── */}
                        <path
                            d="
                              M -20,140
                              L -110,115
                              L -115,160
                              L -28,175
                              Z
                            "
                            fill="#00e5ff"
                            fillOpacity={0.06}
                            stroke="#00e5ff"
                            strokeOpacity={0.5}
                            strokeWidth={0.9}
                        />

                        {/* ── Right engine nozzle ── */}
                        <ellipse
                            cx={12}
                            cy={208}
                            rx={8}
                            ry={12}
                            fill="#00e5ff"
                            fillOpacity={0.1}
                            stroke="#00e5ff"
                            strokeOpacity={0.55}
                            strokeWidth={1}
                        />

                        {/* ── Left engine nozzle ── */}
                        <ellipse
                            cx={-12}
                            cy={208}
                            rx={8}
                            ry={12}
                            fill="#00e5ff"
                            fillOpacity={0.1}
                            stroke="#00e5ff"
                            strokeOpacity={0.55}
                            strokeWidth={1}
                        />

                        {/* ── Centre fuselage panel lines ── */}
                        <line x1={0} y1={-210} x2={0} y2={215}
                            stroke="#00e5ff" strokeOpacity={0.2} strokeWidth={0.6} strokeDasharray="4 6" />
                        <line x1={-28} y1={-20} x2={28} y2={-20}
                            stroke="#00e5ff" strokeOpacity={0.2} strokeWidth={0.6} />
                        <line x1={-26} y1={70} x2={26} y2={70}
                            stroke="#00e5ff" strokeOpacity={0.2} strokeWidth={0.6} />
                        <line x1={-22} y1={140} x2={22} y2={140}
                            stroke="#00e5ff" strokeOpacity={0.2} strokeWidth={0.6} />
                    </g>

                    {/* Grid */}
                    {gridX.map((x) => (
                        <line
                            key={`x-${x}`}
                            x1={x}
                            y1={0}
                            x2={x}
                            y2={480}
                            stroke="#1f334a"
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
                            stroke="#1f334a"
                            strokeWidth={1}
                        />
                    ))}

                    {/* Optical center guidelines */}
                    <line
                        x1={320}
                        x2={320}
                        y1={0}
                        y2={480}
                        stroke="#35506e"
                        strokeDasharray="6 7"
                        opacity={0.85}
                    />

                    <line
                        x1={0}
                        x2={640}
                        y1={240}
                        y2={240}
                        stroke="#35506e"
                        strokeDasharray="6 7"
                        opacity={0.85}
                    />

                    {/* Aircraft filled surfaces */}
                    {bodyShell && (
                        <polygon
                            points={bodyShell}
                            fill="#cfd9e6"
                            fillOpacity={0.08}
                            stroke="#aebfd1"
                            strokeOpacity={0.15}
                            strokeWidth={1}
                        />
                    )}

                    {rightWing && (
                        <polygon
                            points={rightWing}
                            fill="#d7dee8"
                            fillOpacity={0.12}
                            stroke="#b9c6d4"
                            strokeOpacity={0.35}
                            strokeWidth={1.2}
                        />
                    )}

                    {leftWing && (
                        <polygon
                            points={leftWing}
                            fill="#d7dee8"
                            fillOpacity={0.12}
                            stroke="#b9c6d4"
                            strokeOpacity={0.35}
                            strokeWidth={1.2}
                        />
                    )}

                    {rightTail && (
                        <polygon
                            points={rightTail}
                            fill="#d7dee8"
                            fillOpacity={0.1}
                            stroke="#b9c6d4"
                            strokeOpacity={0.35}
                            strokeWidth={1.1}
                        />
                    )}

                    {leftTail && (
                        <polygon
                            points={leftTail}
                            fill="#d7dee8"
                            fillOpacity={0.1}
                            stroke="#b9c6d4"
                            strokeOpacity={0.35}
                            strokeWidth={1.1}
                        />
                    )}

                    {verticalTail && (
                        <polygon
                            points={verticalTail}
                            fill="#d7dee8"
                            fillOpacity={0.08}
                            stroke="#b9c6d4"
                            strokeOpacity={0.3}
                            strokeWidth={1}
                        />
                    )}

                    {/* Main fuselage spine */}
                    {fuselageSpine && (
                        <>
                            <path
                                d={fuselageSpine}
                                fill="none"
                                stroke="#d9e3ef"
                                strokeOpacity={0.18}
                                strokeWidth={8}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                            <path
                                d={fuselageSpine}
                                fill="none"
                                stroke="#dce6f1"
                                strokeOpacity={0.9}
                                strokeWidth={2.2}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                filter="url(#cameraGlow)"
                            />
                        </>
                    )}

                    {/* Secondary structural edges */}
                    {AIRCRAFT_EDGES.map(([a, b], index) => {
                        const pa = points[a];
                        const pb = points[b];

                        if (!pa || !pb || !visible[a] || !visible[b]) {
                            return null;
                        }

                        const isPrimary =
                            (a === 0 && b === 1) ||
                            (a === 1 && b === 2) ||
                            (a === 2 && b === 7) ||
                            (a === 7 && b === 11);

                        return (
                            <line
                                key={index}
                                x1={pa[0]}
                                y1={pa[1]}
                                x2={pb[0]}
                                y2={pb[1]}
                                stroke={isPrimary ? "#dce6f1" : "#96a9bf"}
                                strokeWidth={isPrimary ? 1.8 : 1.4}
                                opacity={isPrimary ? 0.92 : 0.6}
                            />
                        );
                    })}

                    {/* Landmark points */}
                    {points.map((point, index) => {
                        if (!point || !visible[index]) return null;

                        const [dx, dy] = LABEL_OFFSETS[index];

                        return (
                            <g key={index}>
                                <circle
                                    cx={point[0]}
                                    cy={point[1]}
                                    r={10}
                                    fill="url(#landmarkHalo)"
                                    opacity={0.55}
                                />
                                <circle
                                    cx={point[0]}
                                    cy={point[1]}
                                    r={4.7}
                                    fill="#f5a86c"
                                    stroke="#ffe3ca"
                                    strokeWidth={1.1}
                                />

                                {/* small label chip */}
                                <g transform={`translate(${point[0] + dx}, ${point[1] + dy})`}>
                                    <rect
                                        x={-4}
                                        y={-12}
                                        rx={6}
                                        ry={6}
                                        width={18}
                                        height={16}
                                        fill="#091a2b"
                                        fillOpacity={0.9}
                                        stroke="#f0a970"
                                        strokeOpacity={0.8}
                                        strokeWidth={0.9}
                                    />
                                    <text
                                        x={5}
                                        y={0}
                                        textAnchor="middle"
                                        fill="#ffc18e"
                                        fontSize={10}
                                        fontWeight={700}
                                        dominantBaseline="middle"
                                    >
                                        {index}
                                    </text>
                                </g>

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
                        fill="#8ea6bd"
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