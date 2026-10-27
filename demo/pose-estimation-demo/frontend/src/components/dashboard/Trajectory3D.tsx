
"use client";

import dynamic from "next/dynamic";
import { Box } from "lucide-react";

import type { Data, Layout } from "plotly.js";

import {
    ESTIMATOR_COLORS,
    ESTIMATOR_NAMES
} from "@/types/telemetry";

import type {
    EstimatorId,
    ExperimentTelemetry,
    Vector3
} from "@/types/telemetry";

const Plot = dynamic(
    () => import("react-plotly.js"),
    { ssr: false }
);

interface Props {
    telemetry: ExperimentTelemetry | null;
    frameIndex: number;
}

const ORDER: EstimatorId[] = [
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid"
];

function makeLine(
    name: string,
    points: (Vector3 | null)[],
    color: string,
    width = 5
): Data {
    return {
        type: "scatter3d",
        mode: "lines",
        name,
        x: points.map((p) => p?.[0] ?? null),
        y: points.map((p) => p?.[1] ?? null),
        z: points.map((p) => p?.[2] ?? null),
        line: {
            color,
            width
        },
        connectgaps: false,
        hovertemplate:
            "X: %{x:.2f} m<br>" +
            "Y: %{y:.2f} m<br>" +
            "Z: %{z:.2f} m<extra>%{fullData.name}</extra>"
    } as Data;
}

export default function Trajectory3D({
    telemetry,
    frameIndex
}: Props) {
    const frames = telemetry?.frames.slice(
        0,
        frameIndex + 1
    ) ?? [];

    const traces: Data[] = [];

    if (frames.length) {
        traces.push(
            makeLine(
                "Ground Truth",
                frames.map((frame) =>
                    frame.ground_truth.position
                ),
                "#59d5a3",
                7
            )
        );
    }

    const annPoints = frames.map(
        (frame) => frame.ann?.coarse_position ?? null
    );

    if (annPoints.some(Boolean)) {
        traces.push(
            makeLine("ANN Coarse", annPoints, "#ffad64", 4)
        );
    }

    for (const id of ORDER) {
        const points = frames.map(
            (frame) => frame.estimates[id]?.position ?? null
        );

        if (!points.some(Boolean)) continue;

        traces.push(
            makeLine(
                ESTIMATOR_NAMES[id],
                points,
                ESTIMATOR_COLORS[id],
                5
            )
        );
    }

    const currentFrame = frames.at(-1);

    const hybridDiagnostics =
        currentFrame?.diagnostics.hybrid?.pf_diagnostics;

    const hybridParticleSample =
        typeof hybridDiagnostics === "object" &&
        hybridDiagnostics !== null &&
        "particle_sample" in hybridDiagnostics
            ? (hybridDiagnostics as {
                particle_sample?: number[][] | null;
            }).particle_sample
            : null;

    const particleSource =
        currentFrame?.diagnostics.improved_pf?.particle_sample ??
        currentFrame?.diagnostics.paper_pf?.particle_sample ??
        hybridParticleSample ??
        null;

    const particlePoints: number[][] =
        Array.isArray(particleSource)
            ? particleSource.filter(
                (point): point is number[] =>
                    Array.isArray(point) &&
                    point.length === 3 &&
                    point.every(
                        (value) =>
                            typeof value === "number" &&
                            Number.isFinite(value)
                    )
            )
            : [];

    if (particlePoints.length) {
        traces.push({
            type: "scatter3d",
            mode: "markers",
            name: "Particle Cloud",
            x: particlePoints.map((p) => p[0]),
            y: particlePoints.map((p) => p[1]),
            z: particlePoints.map((p) => p[2]),
            marker: {
                color: "#79a9ff",
                size: 2.5,
                opacity: 0.32
            },
            hoverinfo: "skip"
        } as Data);
    }

    const layout: Partial<Layout> = {
        autosize: true,
        margin: { l: 0, r: 0, b: 0, t: 12 },
        paper_bgcolor: "#111b29",
        plot_bgcolor: "#111b29",
        font: {
            color: "#a9bdd0",
            size: 11
        },
        showlegend: true,
        legend: {
            orientation: "h",
            x: 0.02,
            y: 1.05,
            font: { size: 10 }
        },
        scene: {
            bgcolor: "#111b29",
            xaxis: {
                title: { text: "Follower Forward (m)" },
                gridcolor: "#304257",
                zerolinecolor: "#415770"
            },
            yaxis: {
                title: { text: "Follower Right (m)" },
                gridcolor: "#304257",
                zerolinecolor: "#415770"
            },
            zaxis: {
                title: { text: "Follower Down (m)" },
                gridcolor: "#304257",
                zerolinecolor: "#415770"
            },
            aspectmode: "data",
            camera: {
                eye: { x: 1.6, y: 1.6, z: 0.9 },
                center: { x: 0, y: 0, z: 0 },
                up: { x: 0, y: 0, z: 1 }
            }
        }
    };

    return (
        <section className="dashboard-panel trajectory-panel">
            <div className="panel-heading">
                <div>
                    <span className="eyebrow">SPATIAL MAPPING</span>
                    <h2>3D Relative Trajectory</h2>
                </div>
                <Box size={19} />
            </div>

            <div className="plot-container plot-3d">
                {telemetry ? (
                    <Plot
                        data={traces}
                        layout={layout}
                        config={{
                            responsive: true,
                            displaylogo: false,
                            scrollZoom: true
                        }}
                        useResizeHandler
                        style={{ width: "100%", height: "100%" }}
                    />
                ) : (
                    <div className="empty-panel">
                        Run an experiment to display trajectories
                    </div>
                )}
            </div>
        </section>
    );
}
