import abc
import logging
import math
import typing
import skimage

import numpy as np
import tqdm


class BaseSegmentation(abc.ABC):
    def __init__(self, min_area: int, max_area: int = math.inf, silent: bool = False):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area
        self.silent = silent

        if self.max_area is None:
            self.max_area = math.inf

    def get_algorithm_name(self) -> str:
        return self.__class__.__name__

    def segment(
            self, image: np.ndarray, mask: np.ndarray = None, offset: int = 0
    ) -> typing.List[np.ndarray]:

        if mask is None:
            mask = np.ones(image.shape[:2])

        segments = self._segment(image, mask)

        if self.silent:
            iterator = segments

        else:
            logging.info("Found {} initial segments".format(len(segments)))

            logging.info("Removing invalid segments...")
            iterator = tqdm.tqdm(segments)

        valid_segments = []
        for segment in iterator:
            if np.any(segment[:, :offset + 1]) and np.any(segment[:, -(offset + 1):]):
                continue

            if np.any(segment[:offset + 1]) and np.any(segment[-(offset + 1):]):
                continue

            area = segment.sum()

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            valid_segments.append(segment)

        if not self.silent:
            logging.info("{} valid segments remain".format(len(valid_segments)))
            logging.info("Sorting masks by size...")

        valid_segments.sort(key=lambda x: x.sum(), reverse=True)

        return valid_segments

    @abc.abstractmethod
    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        pass

    def _masks_from_binary(self, segmented: np.ndarray, mask: np.ndarray):
        segmented[np.logical_not(mask)] = 0

        labeled = skimage.measure.label(segmented)

        result = []
        for segment in np.unique(labeled):
            if segment == 0:
                continue

            result.append(labeled == segment)

        return result
