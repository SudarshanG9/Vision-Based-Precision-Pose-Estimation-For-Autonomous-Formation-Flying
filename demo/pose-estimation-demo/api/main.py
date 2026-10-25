
"""FastAPI application for the pose estimation research platform.

Run from pose-estimation-demo/:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.scenarios import (
    router as scenarios_router,
)
from api.routes.experiments import (
    router as experiments_router,
)
from api.routes.telemetry import (
    router as telemetry_router,
)

from services.model_registry import ModelRegistry
from services.experiment_store import ExperimentStore
from services.experiment_runner import ExperimentRunner


def create_app(
    model_registry=None,
    experiment_store=None,
    experiment_runner=None,
):
    app = FastAPI(
        title="Vision-Based Pose Estimation API",
        description=(
            "Experiment execution and telemetry for "
            "autonomous formation-flight pose estimation."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    models = (
        model_registry
        if model_registry is not None
        else ModelRegistry(
            checkpoint_path=os.getenv(
                "POSE_MODEL_CHECKPOINT",
                "checkpoints/pose_classifier_best.pt",
            ),
            calibration_path=os.getenv(
                "POSE_CALIBRATION_PATH",
                "checkpoints/classifier_error_calibration.npz",
            ),
            device=os.getenv(
                "POSE_MODEL_DEVICE",
                "auto",
            ),
        )
    )

    store = (
        experiment_store
        if experiment_store is not None
        else ExperimentStore(
            directory=os.getenv(
                "POSE_EXPERIMENT_DIRECTORY",
                "outputs/experiments",
            )
        )
    )

    runner = (
        experiment_runner
        if experiment_runner is not None
        else ExperimentRunner(models, store)
    )

    app.state.model_registry = models
    app.state.experiment_store = store
    app.state.experiment_runner = runner

    app.include_router(scenarios_router)
    app.include_router(experiments_router)
    app.include_router(telemetry_router)

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "healthy",
            "api_version": "v1",
            "models_loaded": models.loaded,
        }

    return app


app = create_app()
