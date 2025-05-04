import typing

import numpy as np
import numba

import common


def interpolate(points, colors, density, position: np.ndarray, p: float = 3) -> np.ndarray:
    distances = np.linalg.norm(points - position, axis=1)
    weights = 1 / (distances + (density * 0.25)) ** p

    numerator = np.sum(weights.reshape((-1, 1)) * colors, axis=0)
    denominator = np.sum(weights)

    return numerator / denominator


@numba.jit(parallel=True, nopython=True)
def rasterize(points, colors, density, size: tuple, p: float = 3, tile_size: int = 64) -> np.ndarray:
    width, height = size
    out_image = np.zeros((height, width, 3), dtype=np.float32)

    n_points = points.shape[0]
    d0 = density * 0.25

    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile_h = y1 - y0
            tile_w = x1 - x0

            for yi in range(tile_h):
                for xi in range(tile_w):
                    y = y0 + yi
                    x = x0 + xi

                    weights_sum = 0.0
                    weights = np.empty(n_points, dtype=np.float32)

                    # Compute weights
                    for idx in range(n_points):
                        dy = y - points[idx, 0]
                        dx = x - points[idx, 1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        weight = 1.0 / (dist + d0) ** p
                        weights[idx] = weight
                        weights_sum += weight

                    # Normalize and interpolate
                    r, g, b = 0.0, 0.0, 0.0
                    for idx in range(n_points):
                        w = weights[idx] / (weights_sum + 1e-8)
                        r += w * colors[idx, 0]
                        g += w * colors[idx, 1]
                        b += w * colors[idx, 2]

                    out_image[y, x, 0] = r
                    out_image[y, x, 1] = g
                    out_image[y, x, 2] = b

    return out_image


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
