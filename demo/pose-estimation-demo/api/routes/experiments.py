from fastapi import APIRouter
from api.schemas import ExperimentRequest
from services.experiment_runner import ExperimentRunner

router=APIRouter(prefix='/experiments', tags=['experiments'])
@router.post('/run')
def run(request: ExperimentRequest): return ExperimentRunner().run(request.steps)
