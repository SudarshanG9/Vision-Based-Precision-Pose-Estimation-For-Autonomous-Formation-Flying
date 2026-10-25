from pydantic import BaseModel
class ExperimentRequest(BaseModel):
    steps: int = 10
