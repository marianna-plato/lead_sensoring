"""Vector and geometric helper functions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle between two vectors in degrees."""
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    cos_angle = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def gaussian_score(value: float, target: float, sigma: float) -> float:
    """Return a smooth score between 0 and 1 based on distance from a target."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(math.exp(-((value - target) / sigma) ** 2))


def mean_coord(coords: Iterable[np.ndarray]) -> np.ndarray:
    """Return the coordinate mean from an iterable of coordinates."""
    array = np.array(list(coords), dtype=float)
    if len(array) == 0:
        raise ValueError("Cannot calculate a mean coordinate from an empty list")
    return np.mean(array, axis=0)


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate approximately uniform unit vectors on a sphere."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    vectors = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n_points):
        y = 1.0 - (2.0 * i / float(n_points - 1))
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        vectors.append([x, y, z])

    return np.array(vectors, dtype=float)


def round_vector(vector: np.ndarray, ndigits: int = 3) -> list[float]:
    """Round a vector into a JSON-serializable list."""
    return [round(float(x), ndigits) for x in vector]
