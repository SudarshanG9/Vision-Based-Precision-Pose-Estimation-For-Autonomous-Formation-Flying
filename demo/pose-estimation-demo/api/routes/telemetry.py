
"""Experiment telemetry and summary endpoints."""

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(
    prefix="/api/v1/experiments",
    tags=["Telemetry"],
)


def require_completed_record(
    experiment_id: str,
    request: Request,
):
    store = request.app.state.experiment_store

    try:
        record = store.get(experiment_id)
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    if record["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "Experiment results are unavailable "
                f"while status is {record['status']}"
            ),
        )

    return record


@router.get("/{experiment_id}/telemetry")
def get_telemetry(
    experiment_id: str,
    request: Request,
):
    record = require_completed_record(
        experiment_id, request
    )

    return record["telemetry"]


@router.get("/{experiment_id}/summary")
def get_summary(
    experiment_id: str,
    request: Request,
):
    record = require_completed_record(
        experiment_id, request
    )

    return {
        "experiment_id": experiment_id,
        **record["summary"],
    }
