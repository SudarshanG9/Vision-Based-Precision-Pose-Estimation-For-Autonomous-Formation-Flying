
"""Consolidated Phase 7 API tests.

Run:
    python -m pytest tests/test_api.py -v
"""

import json
import uuid

import numpy as np
import pytest

from fastapi.testclient import TestClient

from api.main import create_app

from services.experiment_store import ExperimentStore
from services.telemetry_builder import (
    build_telemetry,
    json_safe,
)


class FakeModelRegistry:
    """Avoid requiring a trained checkpoint in API unit tests."""

    loaded = False


class FakeExperimentRunner:
    """Simulate deterministic completion of an experiment."""

    def __init__(self, store):
        self.store = store

    def execute(self, experiment_id):
        self.store.update_status(
            experiment_id,
            "running",
            progress=0.5,
        )

        telemetry = {
            "experiment_id": experiment_id,
            "scenario_id": "distribution_compatible_approach",
            "simulation_fps": 20,
            "playback_fps": 30,
            "frame_count": 1,
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "observation": {
                        "keypoints": [
                            [320.0, 240.0]
                            for _ in range(14)
                        ],
                        "visible": [True] * 14,
                    },
                    "ground_truth": {
                        "position": [35.0, 0.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "ann": {
                        "class_id": 1,
                        "confidence": 0.5,
                        "coarse_position": [35.0, 0.0, 0.0],
                        "coarse_roll_rad": 0.0,
                    },
                    "estimates": {
                        "geometric": {
                            "position": [35.0, 0.0, 0.0],
                            "quaternion": [0.0, 0.0, 0.0, 1.0],
                            "frame_from": "L",
                            "frame_to": "F",
                        }
                    },
                    "errors": {
                        "geometric": {
                            "position_error_m": 0.0,
                            "attitude_error_deg": 0.0,
                            "reprojection_error_px": 0.0,
                        }
                    },
                    "diagnostics": {},
                }
            ],
        }

        summary = {
            "metrics": {
                "geometric": {
                    "frame_count": 1,
                    "failure_rate": 0.0,
                }
            },
            "runtime": {},
            "metadata": {"seed": 42},
        }

        self.store.complete(
            experiment_id,
            summary=summary,
            telemetry=telemetry,
        )


@pytest.fixture
def api_client(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")

    models = FakeModelRegistry()
    runner = FakeExperimentRunner(store)

    app = create_app(
        model_registry=models,
        experiment_store=store,
        experiment_runner=runner,
    )

    with TestClient(app) as client:
        yield client, store


def experiment_payload():
    return {
        "scenario_id": "distribution_compatible_approach",
        "estimators": ["geometric"],
        "parameters": {
            "particle_count": 100,
            "resampling_alpha": 0.9,
            "pixel_noise_std": 1.5,
            "seed": 42,
            "duration_seconds": 1.0,
            "simulation_fps": 10,
        },
    }


def test_health(api_client):
    client, _ = api_client

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["api_version"] == "v1"
    assert result["models_loaded"] is False


def test_scenario_discovery(api_client):
    client, _ = api_client

    response = client.get("/api/v1/scenarios")

    assert response.status_code == 200

    scenarios = response.json()["scenarios"]

    assert len(scenarios) >= 1

    assert scenarios[0]["id"] == (
        "distribution_compatible_approach"
    )


def test_estimator_discovery(api_client):
    client, _ = api_client

    response = client.get("/api/v1/estimators")

    assert response.status_code == 200

    identifiers = {
        item["id"]
        for item in response.json()["estimators"]
    }

    assert identifiers == {
        "paper_pf",
        "improved_pf",
        "geometric",
        "hybrid",
    }


def test_create_and_complete_experiment(api_client):
    client, store = api_client

    response = client.post(
        "/api/v1/experiments",
        json=experiment_payload(),
    )

    assert response.status_code == 202

    data = response.json()
    experiment_id = data["experiment_id"]

    uuid.UUID(experiment_id)

    # TestClient runs the background task before returning.
    stored = store.get(experiment_id)

    assert stored["status"] == "completed"
    assert stored["progress"] == 1.0


def test_experiment_status(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/experiments",
        json=experiment_payload(),
    )

    experiment_id = response.json()["experiment_id"]

    status_response = client.get(
        f"/api/v1/experiments/{experiment_id}"
    )

    assert status_response.status_code == 200

    status_data = status_response.json()

    assert status_data["experiment_id"] == experiment_id
    assert status_data["status"] == "completed"


def test_telemetry_and_summary(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/experiments",
        json=experiment_payload(),
    )

    experiment_id = response.json()["experiment_id"]

    telemetry_response = client.get(
        f"/api/v1/experiments/{experiment_id}/telemetry"
    )

    assert telemetry_response.status_code == 200

    telemetry = telemetry_response.json()

    assert telemetry["frame_count"] == 1
    assert len(telemetry["frames"]) == 1

    frame = telemetry["frames"][0]

    assert len(frame["observation"]["keypoints"]) == 14

    summary_response = client.get(
        f"/api/v1/experiments/{experiment_id}/summary"
    )

    assert summary_response.status_code == 200
    assert "metrics" in summary_response.json()


def test_invalid_request_rejected(api_client):
    client, _ = api_client

    payload = experiment_payload()
    payload["parameters"]["particle_count"] = -1

    response = client.post(
        "/api/v1/experiments",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_estimator_rejected(api_client):
    client, _ = api_client

    payload = experiment_payload()
    payload["estimators"] = ["unknown_estimator"]

    response = client.post(
        "/api/v1/experiments",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_experiment(api_client):
    client, _ = api_client

    unknown_id = str(uuid.uuid4())

    response = client.get(
        f"/api/v1/experiments/{unknown_id}"
    )

    assert response.status_code == 404


def test_pending_telemetry_unavailable(api_client):
    client, store = api_client

    experiment_id = store.create(
        experiment_payload()
    )

    response = client.get(
        f"/api/v1/experiments/{experiment_id}/telemetry"
    )

    assert response.status_code == 409


def test_completed_results_are_immutable(api_client):
    _, store = api_client

    experiment_id = store.create(
        experiment_payload()
    )

    store.update_status(
        experiment_id,
        "running",
        progress=0.5,
    )

    store.complete(
        experiment_id,
        summary={"metrics": {}},
        telemetry={"frames": []},
    )

    with pytest.raises(RuntimeError):
        store.complete(
            experiment_id,
            summary={"metrics": {"changed": True}},
            telemetry={"frames": []},
        )


def test_json_safe_serialization():
    original = {
        "array": np.array([1.0, 2.0]),
        "missing": float("nan"),
        "infinite": float("inf"),
        "count": np.int64(14),
        "visible": np.bool_(True),
    }

    converted = json_safe(original)

    assert converted["array"] == [1.0, 2.0]
    assert converted["missing"] is None
    assert converted["infinite"] is None
    assert converted["count"] == 14
    assert converted["visible"] is True

    json.dumps(converted, allow_nan=False)


def test_telemetry_alignment():
    benchmark = {
        "metadata": {
            "scenario": "distribution_compatible_approach",
            "simulation_fps": 20,
            "frame_count": 1,
        },
        "ground_truth": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "position": [35.0, 0.0, 0.0],
                "quaternion": [0, 0, 0, 1],
            }
        ],
        "observations": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "keypoints": [None] * 14,
                "visible": [False] * 14,
            }
        ],
        "results": {
            "geometric": {
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "estimate": None,
                        "success": False,
                        "position_error_m": None,
                        "attitude_error_deg": None,
                        "reprojection_error_px": None,
                        "runtime_ms": 1.0,
                        "diagnostics": {},
                        "failure_reason": "No landmarks",
                    }
                ]
            }
        },
    }

    telemetry = build_telemetry(
        "test_experiment",
        benchmark,
    )

    assert telemetry["frame_count"] == 1

    frame = telemetry["frames"][0]

    assert frame["estimates"]["geometric"] is None

    assert (
        frame["diagnostics"]["geometric"]["success"]
        is False
    )
