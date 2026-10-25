
"""Experiment creation and status endpoints."""

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    status,
)

from api.schemas import (
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentStatusResponse,
)


router = APIRouter(
    prefix="/api/v1/experiments",
    tags=["Experiments"],
)


def execute_background(runner, experiment_id):
    """Run a scheduled experiment; failures are stored by the runner."""
    try:
        runner.execute(experiment_id)
    except Exception:
        # The runner records the failure in ExperimentStore.
        # Avoid crashing the HTTP response after it was sent.
        pass


@router.post(
    "",
    response_model=ExperimentCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_experiment(
    payload: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    store = request.app.state.experiment_store
    runner = request.app.state.experiment_runner

    parameters = payload.parameters.model_dump()

    configuration = {
        "scenario_id": payload.scenario_id,
        "estimators": payload.estimators,
        **parameters,
    }

    experiment_id = store.create(configuration)

    background_tasks.add_task(
        execute_background,
        runner,
        experiment_id,
    )

    return ExperimentCreateResponse(
        experiment_id=experiment_id,
        status="pending",
    )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentStatusResponse,
)
def get_experiment(
    experiment_id: str,
    request: Request,
):
    store = request.app.state.experiment_store

    try:
        record = store.status(experiment_id)
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return ExperimentStatusResponse(**record)
