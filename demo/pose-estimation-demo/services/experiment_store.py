
"""Persistent experiment metadata and immutable results."""

from __future__ import annotations

import json
import os
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict):
    """Replace a JSON document atomically within its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name

            json.dump(
                data,
                handle,
                indent=2,
                allow_nan=False,
            )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)

    finally:
        if (
            temporary_name is not None
            and os.path.exists(temporary_name)
        ):
            os.unlink(temporary_name)


class ExperimentStore:
    STATUSES = {
        "pending",
        "running",
        "completed",
        "failed",
    }

    def __init__(self, directory="outputs/experiments"):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True, exist_ok=True
        )

        self._lock = RLock()

    def _validate_id(self, experiment_id: str):
        try:
            normalized = str(uuid.UUID(experiment_id))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid experiment identifier")

        if normalized != experiment_id:
            raise ValueError("Noncanonical experiment identifier")

    def _path(self, experiment_id):
        self._validate_id(experiment_id)

        return self.directory / f"{experiment_id}.json"

    def create(self, configuration: dict):
        with self._lock:
            experiment_id = str(uuid.uuid4())

            record = {
                "experiment_id": experiment_id,
                "status": "pending",
                "progress": 0.0,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "configuration": configuration,
                "error": None,
                "summary": None,
                "telemetry": None,
            }

            atomic_write_json(
                self._path(experiment_id),
                record,
            )

            return experiment_id

    def get(self, experiment_id):
        path = self._path(experiment_id)

        with self._lock:
            if not path.is_file():
                raise KeyError(experiment_id)

            with path.open(
                "r", encoding="utf-8"
            ) as handle:
                return json.load(handle)

    def update_status(
        self,
        experiment_id,
        status,
        progress=None,
        error=None,
    ):
        if status not in self.STATUSES:
            raise ValueError("Invalid experiment status")

        with self._lock:
            record = self.get(experiment_id)

            if record["status"] in ("completed", "failed"):
                raise RuntimeError(
                    "Terminal experiment cannot be modified"
                )

            allowed = {
                "pending": {"running", "failed"},
                "running": {"running", "failed"},
            }

            if status not in allowed.get(record["status"], set()):
                raise ValueError(
                    "Invalid experiment status transition"
                )

            record["status"] = status
            record["updated_at"] = utc_now()

            if progress is not None:
                if not 0.0 <= progress <= 1.0:
                    raise ValueError(
                        "progress must be in [0,1]"
                    )
                record["progress"] = float(progress)

            if status == "failed":
                record["error"] = (
                    str(error) if error is not None
                    else "Experiment execution failed"
                )

            atomic_write_json(
                self._path(experiment_id),
                record,
            )

    def complete(
        self,
        experiment_id,
        summary: dict,
        telemetry: dict,
    ):
        with self._lock:
            record = self.get(experiment_id)

            if record["status"] != "running":
                raise RuntimeError(
                    "Only running experiments may be completed"
                )

            record["status"] = "completed"
            record["progress"] = 1.0
            record["updated_at"] = utc_now()
            record["summary"] = summary
            record["telemetry"] = telemetry

            atomic_write_json(
                self._path(experiment_id),
                record,
            )

    def status(self, experiment_id):
        record = self.get(experiment_id)

        return {
            "experiment_id": record["experiment_id"],
            "status": record["status"],
            "progress": record["progress"],
            "error": record["error"],
        }

    def telemetry(self, experiment_id):
        record = self.get(experiment_id)

        if record["status"] != "completed":
            raise RuntimeError(
                "Experiment has not completed"
            )

        return record["telemetry"]

    def summary(self, experiment_id):
        record = self.get(experiment_id)

        if record["status"] != "completed":
            raise RuntimeError(
                "Experiment has not completed"
            )

        return record["summary"]
