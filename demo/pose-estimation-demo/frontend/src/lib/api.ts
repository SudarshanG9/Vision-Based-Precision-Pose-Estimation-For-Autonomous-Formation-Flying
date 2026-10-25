
import type {
    EstimatorInfo,
    ExperimentCreated,
    ExperimentProgress,
    ExperimentRequest,
    ExperimentSummary,
    ExperimentTelemetry,
    ScenarioInfo
} from "@/types/telemetry";

const API_BASE =
    process.env.NEXT_PUBLIC_API_URL ??
    "http://127.0.0.1:8000/api/v1";

export class ApiError extends Error {
    constructor(
        message: string,
        public status: number
    ) {
        super(message);
        this.name = "ApiError";
    }
}

async function request<T>(
    path: string,
    init?: RequestInit
): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
        ...init,
        cache: "no-store",
        headers: {
            "Content-Type": "application/json",
            ...init?.headers
        }
    });

    if (!response.ok) {
        let message = `API request failed (${response.status})`;

        try {
            const body: unknown = await response.json();

            if (typeof body === "object" && body !== null) {
                const detail = (body as { detail?: unknown }).detail;

                if (typeof detail === "string") {
                    message = detail;
                } else if (detail !== undefined) {
                    message = JSON.stringify(detail);
                }
            }
        } catch {
            // Preserve the HTTP status message.
        }

        throw new ApiError(message, response.status);
    }

    return response.json() as Promise<T>;
}

export const api = {
    health: () =>
        request<{
            status: string;
            api_version: string;
            models_loaded: boolean;
        }>("/health"),

    scenarios: async () => {
        const result = await request<{
            scenarios: ScenarioInfo[];
        }>("/scenarios");

        return result.scenarios;
    },

    estimators: async () => {
        const result = await request<{
            estimators: EstimatorInfo[];
        }>("/estimators");

        return result.estimators;
    },

    createExperiment: (payload: ExperimentRequest) =>
        request<ExperimentCreated>("/experiments", {
            method: "POST",
            body: JSON.stringify(payload)
        }),

    experimentStatus: (id: string) =>
        request<ExperimentProgress>(
            `/experiments/${encodeURIComponent(id)}`
        ),

    telemetry: (id: string) =>
        request<ExperimentTelemetry>(
            `/experiments/${encodeURIComponent(id)}/telemetry`
        ),

    summary: (id: string) =>
        request<ExperimentSummary>(
            `/experiments/${encodeURIComponent(id)}/summary`
        )
};
