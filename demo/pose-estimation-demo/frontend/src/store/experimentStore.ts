
import { create } from "zustand";

import type {
    EstimatorId,
    ExperimentParameters,
    ExperimentProgress,
    ExperimentSummary,
    ExperimentTelemetry
} from "@/types/telemetry";

export const DEFAULT_PARAMETERS: ExperimentParameters = {
    particle_count: 500,
    resampling_alpha: 0.9,
    pixel_noise_std: 1.5,
    seed: 42,
    duration_seconds: 2,
    simulation_fps: 20,
    position_noise_std: 0.5,
    rotation_noise_std: 0.01,
    measurement_sigma_px: 2,
    ess_threshold_ratio: 0.5
};

export const DEFAULT_ESTIMATORS: EstimatorId[] = [
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid"
];

interface ExperimentState {
    scenarioId: string;
    estimators: EstimatorId[];
    parameters: ExperimentParameters;

    experimentId: string | null;
    progress: ExperimentProgress | null;
    telemetry: ExperimentTelemetry | null;
    summary: ExperimentSummary | null;

    isSubmitting: boolean;
    error: string | null;

    setScenario: (value: string) => void;
    toggleEstimator: (id: EstimatorId) => void;
    setParameter: <K extends keyof ExperimentParameters>(
        key: K,
        value: ExperimentParameters[K]
    ) => void;

    begin: () => void;
    setExperimentId: (id: string) => void;
    setProgress: (value: ExperimentProgress) => void;
    setResults: (
        telemetry: ExperimentTelemetry,
        summary: ExperimentSummary
    ) => void;
    setError: (message: string) => void;
}

export const useExperimentStore = create<ExperimentState>(
    (set) => ({
        scenarioId: "distribution_compatible_approach",
        estimators: [...DEFAULT_ESTIMATORS],
        parameters: { ...DEFAULT_PARAMETERS },

        experimentId: null,
        progress: null,
        telemetry: null,
        summary: null,

        isSubmitting: false,
        error: null,

        setScenario: (scenarioId) => set({ scenarioId }),

        toggleEstimator: (id) =>
            set((state) => {
                const exists = state.estimators.includes(id);

                return {
                    estimators: exists
                        ? state.estimators.filter((x) => x !== id)
                        : [...state.estimators, id]
                };
            }),

        setParameter: (key, value) =>
            set((state) => ({
                parameters: {
                    ...state.parameters,
                    [key]: value
                }
            })),

        begin: () =>
            set({
                experimentId: null,
                progress: null,
                telemetry: null,
                summary: null,
                isSubmitting: true,
                error: null
            }),

        setExperimentId: (experimentId) =>
            set({
                experimentId,
                isSubmitting: false
            }),

        setProgress: (progress) => set({ progress }),

        setResults: (telemetry, summary) =>
            set({
                telemetry,
                summary,
                isSubmitting: false,
                error: null
            }),

        setError: (error) =>
            set({
                error,
                isSubmitting: false
            })
    })
);
