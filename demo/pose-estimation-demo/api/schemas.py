
"""Pydantic request/response contracts for API v1."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EstimatorName = Literal[
    "paper_pf",
    "improved_pf",
    "geometric",
    "hybrid",
]


class ExperimentParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    particle_count: int = Field(default=1000, ge=1, le=50000)
    resampling_alpha: float = Field(default=0.9, ge=0, le=1)

    pixel_noise_std: float = Field(default=1.5, ge=0, le=50)
    seed: int = Field(default=42, ge=0)

    duration_seconds: float = Field(default=2.0, gt=0, le=60)
    simulation_fps: int = Field(default=20, ge=2, le=200)

    position_noise_std: float = Field(default=0.5, ge=0)
    rotation_noise_std: float = Field(default=0.01, ge=0)

    measurement_sigma_px: float = Field(
        default=2.0, gt=0
    )
    ess_threshold_ratio: float = Field(
        default=0.5, gt=0, le=1
    )


class ExperimentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: Literal[
        "distribution_compatible_approach"
    ] = "distribution_compatible_approach"

    estimators: list[EstimatorName] = Field(
        default_factory=lambda: [
            "paper_pf",
            "improved_pf",
            "geometric",
            "hybrid",
        ],
        min_length=1,
    )

    parameters: ExperimentParameters = Field(
        default_factory=ExperimentParameters
    )

    @model_validator(mode="after")
    def check_estimator_names(self):
        if len(set(self.estimators)) != len(self.estimators):
            raise ValueError(
                "Estimator names must be unique"
            )
        return self


class ExperimentCreateResponse(BaseModel):
    experiment_id: str
    status: Literal["pending"]


class ExperimentStatusResponse(BaseModel):
    experiment_id: str

    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
    ]

    progress: float = Field(ge=0, le=1)
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    api_version: str
    models_loaded: bool


class ScenarioInfo(BaseModel):
    id: str
    name: str
    duration_seconds: float
    simulation_fps: int
    description: str


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfo]


class EstimatorInfo(BaseModel):
    id: str
    name: str
    output_type: str


class EstimatorListResponse(BaseModel):
    estimators: list[EstimatorInfo]
