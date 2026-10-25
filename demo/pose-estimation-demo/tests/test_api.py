from fastapi.testclient import TestClient
from api.main import app

def test_api(): assert TestClient(app).post('/experiments/run',json={'steps':2}).status_code == 200
