import typing

import numpy as np

import common


def interpolate(points, colors, density, position: np.ndarray, p: float = 3) -> np.ndarray:
    distances = np.linalg.norm(points - position, axis=1)
    weights = 1 / (distances + (density * 0.25)) ** p

    numerator = np.sum(weights.reshape((-1, 1)) * colors, axis=0)
    denominator = np.sum(weights)

    return numerator / denominator


def rasterize(points, colors, density, size: tuple, p: float = 3) -> np.ndarray:
    width, height = size

    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))  # (height, width)

    pixel_positions = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)  # (height*width, 2)

    distances = np.linalg.norm(pixel_positions[:, np.newaxis, :] - points, axis=-1)  # (height*width, N)

    weights = 1 / (distances + (density * 0.25)) ** p

    weights_sum = np.sum(weights, axis=1, keepdims=True)  # (height*width, 1)
    normalized_weights = weights / weights_sum  # (height*width, N)

    interpolated_r = np.sum(normalized_weights * colors[:, 0], axis=1)
    interpolated_g = np.sum(normalized_weights * colors[:, 1], axis=1)
    interpolated_b = np.sum(normalized_weights * colors[:, 2], axis=1)

    interpolated_image = np.stack([interpolated_r, interpolated_g, interpolated_b], axis=-1)
    interpolated_image = interpolated_image.reshape(height, width, 3)

    return interpolated_image


class GradientField(common.SavableObject):
    def __init__(
            self, points: np.ndarray, colors: np.ndarray, density: float,
            raster: np.ndarray, size: typing.Tuple[int, int]
    ):
        self.points = points
        self.colors = colors
        self.raster = raster
        self.density = density
        self.size = size

