import typing

import bridson
import numpy as np


def generate_gradient_field(colors: np.ndarray, density, size: typing.Tuple[int, int]):
    control_points = bridson.poisson_disc_samples(*size, density)
    control_points = np.array([(point[1], point[0]) for point in control_points]).astype(int)

    control_colors = colors[np.random.randint(0, len(colors), len(control_points))]

    return control_points, control_colors
