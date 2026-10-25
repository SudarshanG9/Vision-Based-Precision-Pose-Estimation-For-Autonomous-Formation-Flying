
"""Simplified rigid aircraft geometry with 14 landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.interfaces import NUM_LANDMARKS


LANDMARK_NAMES = (
    "nose",
    "cockpit",
    "fuselage_front",
    "right_wing_root",
    "right_wing_tip",
    "left_wing_root",
    "left_wing_tip",
    "fuselage_rear",
    "right_tail_tip",
    "left_tail_tip",
    "vertical_tail_top",
    "tail_end",
    "right_wing_trailing",
    "left_wing_trailing",
)

# FRD coordinates: x forward, y right, z down.
# All dimensions are illustrative and measured in meters.
LANDMARK_COORDINATES = np.array([
    [ 8.0,  0.0,  0.0],
    [ 5.5,  0.0, -0.6],
    [ 3.0,  0.0,  0.0],
    [ 1.0,  1.2,  0.0],
    [-0.8,  9.0,  0.1],
    [ 1.0, -1.2,  0.0],
    [-0.8, -9.0,  0.1],
    [-5.0,  0.0,  0.0],
    [-6.5,  3.6,  0.0],
    [-6.5, -3.6,  0.0],
    [-6.0,  0.0, -2.4],
    [-8.0,  0.0,  0.0],
    [-1.8,  7.5,  0.1],
    [-1.8, -7.5,  0.1],
], dtype=np.float64)

# Wireframe connections used by the dashboard and renderer.
AIRCRAFT_EDGES = (
    (0, 1), (1, 2), (2, 7), (7, 11),
    (2, 3), (3, 4), (4, 12), (12, 3),
    (2, 5), (5, 6), (6, 13), (13, 5),
    (7, 8), (8, 11),
    (7, 9), (9, 11),
    (7, 10), (10, 11),
)

# Opaque surface patches for geometric self-occlusion.
# These are intentionally approximate triangular surfaces.
AIRCRAFT_TRIANGLES = (
    (2, 3, 4),
    (2, 4, 12),
    (2, 5, 6),
    (2, 6, 13),
    (7, 8, 11),
    (7, 9, 11),
    (7, 10, 11),
    (0, 1, 2),
)


@dataclass(frozen=True)
class AircraftModel:
    """Immutable geometry specification."""

    landmarks: np.ndarray
    names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    triangles: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        points = np.asarray(self.landmarks, dtype=np.float64).copy()

        if points.shape != (NUM_LANDMARKS, 3):
            raise ValueError("Expected exactly 14 aircraft landmarks")

        if len(self.names) != NUM_LANDMARKS:
            raise ValueError("Expected 14 landmark names")

        if len(set(self.names)) != NUM_LANDMARKS:
            raise ValueError("Landmark names must be unique")

        if not np.all(np.isfinite(points)):
            raise ValueError("Aircraft geometry must be finite")

        for edge in self.edges:
            if len(edge) != 2 or any(
                not 0 <= i < NUM_LANDMARKS for i in edge
            ):
                raise ValueError("Invalid aircraft edge")

        for triangle in self.triangles:
            if len(triangle) != 3 or any(
                not 0 <= i < NUM_LANDMARKS for i in triangle
            ):
                raise ValueError("Invalid aircraft triangle")

        points.setflags(write=False)
        object.__setattr__(self, "landmarks", points)

    def points(self):
        return self.landmarks.copy()

    def bounding_extent(self):
        return np.ptp(self.landmarks, axis=0)


def create_aircraft_model() -> AircraftModel:
    return AircraftModel(
        landmarks=LANDMARK_COORDINATES,
        names=LANDMARK_NAMES,
        edges=AIRCRAFT_EDGES,
        triangles=AIRCRAFT_TRIANGLES,
    )
