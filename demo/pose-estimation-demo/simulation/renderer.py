
"""Lightweight synthetic wireframe renderer with self-occlusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.aircraft_model import AircraftModel
from simulation.camera import PinholeCamera


@dataclass
class RenderResult:
    image: np.ndarray
    pixels: np.ndarray
    visible: np.ndarray
    camera_points: np.ndarray


def ray_triangle_intersection(
    point,
    triangle,
    epsilon=1e-8,
):
    """
    Test whether a triangle lies between camera origin and point.

    All coordinates are in the optical camera frame.

    Returns True if the landmark is occluded.
    """
    point = np.asarray(point, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)

    distance = np.linalg.norm(point)
    if distance <= epsilon:
        return False

    direction = point / distance
    v0, v1, v2 = triangle

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    if abs(a) < epsilon:
        return False

    f = 1.0 / a
    s = -v0

    u = f * np.dot(s, h)
    if u < -epsilon or u > 1.0 + epsilon:
        return False

    q = np.cross(s, edge1)

    v = f * np.dot(direction, q)
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False

    t = f * np.dot(edge2, q)

    # A surface must be strictly in front of the landmark.
    return epsilon < t < distance - 1e-5


def compute_visibility(
    camera_points,
    projection_valid,
    triangles,
):
    """Apply aircraft self-occlusion to projected landmarks."""
    visible = np.asarray(
        projection_valid, dtype=bool
    ).copy()

    for landmark_id, point in enumerate(camera_points):
        if not visible[landmark_id]:
            continue

        for triangle_ids in triangles:
            if landmark_id in triangle_ids:
                continue

            triangle = camera_points[list(triangle_ids)]

            if ray_triangle_intersection(point, triangle):
                visible[landmark_id] = False
                break

    return visible


def draw_line(image, p0, p1, color):
    """Draw a clipped line segment using NumPy."""
    h, w = image.shape[:2]

    if not (
        np.all(np.isfinite(p0))
        and np.all(np.isfinite(p1))
    ):
        return

    x0, y0 = np.asarray(p0, dtype=np.float64)
    x1, y1 = np.asarray(p1, dtype=np.float64)

    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1

    if steps < 1:
        return

    steps = min(steps, 4 * max(h, w))

    x = np.rint(np.linspace(x0, x1, steps)).astype(int)
    y = np.rint(np.linspace(y0, y1, steps)).astype(int)

    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    image[y[inside], x[inside]] = color


def draw_point(image, pixel, color, radius=3):
    h, w = image.shape[:2]

    if not np.all(np.isfinite(pixel)):
        return

    cx, cy = np.rint(pixel).astype(int)

    x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
    y0, y1 = max(0, cy-radius), min(h, cy+radius+1)

    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx-cx)**2 + (yy-cy)**2 <= radius**2

    patch = image[y0:y1, x0:x1]
    patch[mask] = color


def render_aircraft(
    aircraft: AircraftModel,
    camera: PinholeCamera,
    leader_in_follower,
    draw_hidden=False,
) -> RenderResult:
    """
    Render projected aircraft structure.

    The displayed wireframe is illustrative; geometric feature
    visibility is determined using the triangular surface model.
    """
    pixels, valid, camera_points = camera.project_landmarks(
        aircraft.landmarks,
        leader_in_follower,
    )

    visible = compute_visibility(
        camera_points,
        valid,
        aircraft.triangles,
    )

    height = camera.intrinsics.height
    width = camera.intrinsics.width

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (12, 20, 31)

    # Draw structural edges.
    for start, end in aircraft.edges:
        if not (valid[start] and valid[end]):
            continue

        if not draw_hidden and not (
            visible[start] and visible[end]
        ):
            continue

        draw_line(
            image,
            pixels[start],
            pixels[end],
            color=(104, 147, 189),
        )

    # Draw feature markers.
    for i, pixel in enumerate(pixels):
        if visible[i]:
            draw_point(
                image, pixel,
                color=(75, 225, 165),
                radius=3,
            )
        elif draw_hidden and valid[i]:
            draw_point(
                image, pixel,
                color=(230, 115, 70),
                radius=2,
            )

    return RenderResult(
        image=image,
        pixels=pixels,
        visible=visible,
        camera_points=camera_points,
    )
