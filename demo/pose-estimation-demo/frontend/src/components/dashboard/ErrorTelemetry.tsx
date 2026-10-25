
"use client";

import dynamic from "next/dynamic";
import { Activity } from "lucide-react";

import type { Data, Layout } from "plotly.js";

import {
    ESTIMATOR_COLORS,
    ESTIMATOR_NAMES
} from "@/types/telemetry";

import type {
    EstimatorId,
    ExperimentTelemetry
} from "@/types/telemetry";

const Plot = dynamic(
    () => import("react-plotly.js"),
    { ssr: false }
);

interface Props {
    telemetry: ExperimentTelemetry | null;
    frameIndex: number;
}

const METHODS: EstimatorId[] = [
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid"
];

type ErrorField =
    | "position_error_m"
    | "attitude_error_deg";

function ErrorChart({
    telemetry,
    frameIndex,
    metric,
    title,
    unit
}: {
    telemetry: ExperimentTelemetry | null;
    frameIndex: number;
    metric: ErrorField;
    title: string;
    unit: string;
}) {
    const frames = telemetry?.frames.slice(
        0,
        frameIndex + 1
    ) ?? [];

    const traces: Data[] = METHODS.map((id) => {
        const values = frames.map(
            (frame) => frame.errors[id]?.[metric] ?? null
        );

        if (!values.some((value) => value !== null)) {
            return null;
        }

        return {
            type: "scatter",
            mode: "lines",
            name: ESTIMATOR_NAMES[id],
            x: frames.map((frame) => frame.timestamp),
            y: values,
            line: {
                color: ESTIMATOR_COLORS[id],
                width: 2.5
            },
            connectgaps: false,
            hovertemplate:
                "%{y:.3f} " + unit +
                "<br>Time: %{x:.2f} s" +
                "<extra>%{fullData.name}</extra>"
        } as Data;
    }).filter((trace): trace is Data => trace !== null);

    const layout: Partial<Layout> = {
        autosize: true,
        margin: {
            l: 52,
            r: 14,
            t: 8,
            b: 40
        },
        paper_bgcolor: "#111b29",
        plot_bgcolor: "#111b29",
        font: {
            color: "#9eb1c8",
            size: 11
        },
        xaxis: {
            title: { text: "Simulation Time (s)" },
            gridcolor: "#27384e",
            zeroline: false
        },
        yaxis: {
            title: { text: unit },
            gridcolor: "#27384e",
            zeroline: false,
            rangemode: "tozero"
        },
        legend: {
            orientation: "h",
            x: 0,
            y: 1.2,
            font: { size: 10 }
        },
        showlegend: true,
        hovermode: "x unified"
    };

    return (
        <section className="dashboard-panel error-panel">
            <div className="panel-heading">
                <div>
                    <span className="eyebrow">ERROR ANALYSIS</span>
                    <h2>{title}</h2>
                </div>
                <Activity size={18} />
            </div>

            <div className="plot-container plot-error">
                {telemetry ? (
                    <Plot
                        data={traces}
                        layout={layout}
                        config={{
                            responsive: true,
                            displaylogo: false,
                            displayModeBar: false
                        }}
                        useResizeHandler
                        style={{
                            width: "100%",
                            height: "100%"
                        }}
                    />
                ) : (
                    <div className="empty-panel">
                        Error telemetry unavailable
                    </div>
                )}
            </div>
        </section>
    );
}

export default function ErrorTelemetry({
    telemetry,
    frameIndex
}: Props) {
    return (
        <div className="error-grid">
            <ErrorChart
                telemetry={telemetry}
                frameIndex={frameIndex}
                metric="position_error_m"
                title="Position Estimation Error"
                unit="m"
            />

            <ErrorChart
                telemetry={telemetry}
                frameIndex={frameIndex}
                metric="attitude_error_deg"
                title="Attitude Estimation Error"
                unit="deg"
            />
        </div>
    );
}
