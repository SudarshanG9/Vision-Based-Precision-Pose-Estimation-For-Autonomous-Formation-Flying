
"""Scenario and estimator discovery endpoints."""

from fastapi import APIRouter

from api.schemas import (
    EstimatorInfo,
    EstimatorListResponse,
    ScenarioInfo,
    ScenarioListResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Discovery"],
)


@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
)
def list_scenarios():
    return ScenarioListResponse(
        scenarios=[
            ScenarioInfo(
                id="distribution_compatible_approach",
                name="Nominal Formation Approach",
                duration_seconds=10.0,
                simulation_fps=50,
                description=(
                    "Synthetic leader approach within the "
                    "trained ANN pose distribution."
                ),
            ),
        ]
    )


@router.get(
    "/estimators",
    response_model=EstimatorListResponse,
)
def list_estimators():
    return EstimatorListResponse(
        estimators=[
            EstimatorInfo(
                id="paper_pf",
                name="Original Paper Particle Filter",
                output_type="pose_6dof",
            ),
            EstimatorInfo(
                id="improved_pf",
                name="Improved Particle Filter",
                output_type="pose_6dof",
            ),
            EstimatorInfo(
                id="geometric",
                name="Geometric PnP Refinement",
                output_type="pose_6dof",
            ),
            EstimatorInfo(
                id="hybrid",
                name="Hybrid Pose Estimator",
                output_type="pose_6dof",
            ),
        ]
    )
