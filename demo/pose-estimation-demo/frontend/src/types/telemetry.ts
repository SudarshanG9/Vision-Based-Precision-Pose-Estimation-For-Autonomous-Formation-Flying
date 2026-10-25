
export type Vector2 = [number, number];
export type Vector3 = [number, number, number];
export type Quaternion = [number, number, number, number];

export type EstimatorId =
    | "paper_pf"
    | "improved_pf"
    | "geometric"
    | "hybrid";

export type ExperimentStatus =
    | "pending"
    | "running"
    | "completed"
    | "failed";

export interface PoseTelemetry {
    position: Vector3;
    quaternion: Quaternion;
    frame_from?: string;
    frame_to?: string;
}

export interface CameraFrame {
    keypoints: (Vector2 | null)[];
    visible: boolean[];
}

export interface AnnTelemetry {
    class_id: number;
    confidence: number | null;
    coarse_position: Vector3;
    coarse_roll_rad: number | null;
}

export interface ErrorMetrics {
    position_error_m: number | null;
    attitude_error_deg: number | null;
    reprojection_error_px: number | null;
}

export interface EstimatorDiagnostics {
    runtime_ms?: number | null;
    effective_sample_size?: number | null;
    ess_ratio?: number | null;
    particle_count?: number | null;
    particle_sample?: number[][] | null;
    ann_reseed_performed?: boolean;
    resampling_performed?: boolean;
    selected_estimator?: string;
    visible_features?: number;
    inlier_count?: number;
    success?: boolean;
    failure_reason?: string | null;
    [key: string]: unknown;
}

export interface TelemetryFrame {
    frame_index: number;
    timestamp: number;
    observation: CameraFrame;
    ground_truth: PoseTelemetry;
    ann: AnnTelemetry | null;
    estimates: Partial<
        Record<EstimatorId, PoseTelemetry | null>
    >;
    errors: Partial<Record<EstimatorId, ErrorMetrics>>;
    diagnostics: Partial<
        Record<EstimatorId, EstimatorDiagnostics>
    >;
}

export interface ExperimentTelemetry {
    experiment_id: string;
    scenario_id: string;
    simulation_fps: number;
    playback_fps: number;
    frame_count: number;
    frames: TelemetryFrame[];
}

export interface ExperimentParameters {
    particle_count: number;
    resampling_alpha: number;
    pixel_noise_std: number;
    seed: number;
    duration_seconds: number;
    simulation_fps: number;
    position_noise_std: number;
    rotation_noise_std: number;
    measurement_sigma_px: number;
    ess_threshold_ratio: number;
}

export interface ExperimentRequest {
    scenario_id: string;
    estimators: EstimatorId[];
    parameters: ExperimentParameters;
}

export interface ExperimentCreated {
    experiment_id: string;
    status: "pending";
}

export interface ExperimentProgress {
    experiment_id: string;
    status: ExperimentStatus;
    progress: number;
    error: string | null;
}

export interface StatisticSummary {
    count: number;
    mean: number | null;
    median: number | null;
    std: number | null;
    p95: number | null;
    max: number | null;
}

export interface EstimatorSummary {
    frame_count: number;
    successful_frames: number;
    failed_frames: number;
    failure_rate: number;
    position_error_m: StatisticSummary;
    attitude_error_deg: StatisticSummary;
    runtime_ms: StatisticSummary;
    reprojection_error_px: StatisticSummary;
    attempted_runtime_ms: StatisticSummary;
    mean_estimator_fps: number | null;
}

export interface ExperimentSummary {
    experiment_id: string;
    metrics: Partial<Record<EstimatorId, EstimatorSummary>>;
    runtime: Partial<Record<EstimatorId, StatisticSummary>>;
    metadata: Record<string, unknown>;
}

export interface ScenarioInfo {
    id: string;
    name: string;
    description: string;
    duration_seconds: number;
    simulation_fps: number;
}

export interface EstimatorInfo {
    id: EstimatorId;
    name: string;
    output_type: string;
}

export const ESTIMATOR_NAMES: Record<EstimatorId, string> = {
    paper_pf: "Original PF",
    improved_pf: "Improved PF",
    geometric: "Geometric PnP",
    hybrid: "Hybrid"
};

export const ESTIMATOR_COLORS: Record<EstimatorId, string> = {
    paper_pf: "#ffac69",
    improved_pf: "#bc9cff",
    geometric: "#6cb7ff",
    hybrid: "#f3d16c"
};
