import numpy as np
import tqdm

import segmentation.base_segmentation as base_segmentation


class FloodFillSegmentation(base_segmentation.BaseSegmentation):
    def __init__(self, tolerance: float, min_area: int = 0, max_area: int = None, silent: bool = False):
        super().__init__(min_area, max_area, silent=silent)
        self.tolerance = tolerance

    def flood(self, image: np.ndarray, mask: np.ndarray, seed):
        h, w, _ = image.shape
        filled = np.zeros((h, w), dtype=bool)

        stack = [seed[::-1]]
        seed_color = image[seed[::-1]]

        while stack:
            y, x = stack.pop()

            if not (0 <= y < h and 0 <= x < w):
                continue

            if filled[y, x] or not mask[y, x]:
                continue

            if np.sqrt(np.sum((image[y, x] - seed_color) ** 2)) > self.tolerance:
                continue

            filled[y, x] = True

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = y + dy, x + dx
                stack.append((nx, ny))

        return filled

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        h, w, _ = image.shape
        processed = np.zeros((h, w), dtype=bool)
        remaining_mask = mask.copy()

        if self.silent:
            progress_bar = None
        else:
            progress_bar = tqdm.tqdm(total=len(np.where(remaining_mask == 1)[0]))

        segments = []

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or processed[y, x]:
                    continue

                segment_mask = self.flood(image, remaining_mask, (x, y))

                if np.any(segment_mask):
                    segments.append(segment_mask)
                    processed = np.logical_or(processed, segment_mask)
                    remaining_mask = np.logical_and(remaining_mask, np.logical_not(segment_mask))

                    if progress_bar is not None:
                        progress_bar.update(segment_mask.sum())

        if progress_bar is not None:
            progress_bar.close()

        return segments
