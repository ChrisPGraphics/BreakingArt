import math

import numpy as np
import skimage
from skimage.segmentation import slic

import segmentation.base_segmentation as base_segmentation


class SLICSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, segment_count: int = 100, compactness: float = 10, max_iterations: float = 10, sigma: float = 0,
            min_size_factor: float = 0.5, max_size_factor: float = 3, min_area: int = 0,
            max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(min_area, max_area, silent)

        self.segment_count = segment_count
        self.compactness = compactness
        self.max_iterations = max_iterations
        self.sigma = sigma
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        segments = slic(
            image, n_segments=self.segment_count, compactness=self.compactness, max_num_iter=self.max_iterations,
            sigma=self.sigma, min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor
        )

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        return result
