import math

import numpy as np
import tqdm

import segmentation.base_segmentation as base_segmentation


class IrregularPatchSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, mean_area: float, std_area: float,
            min_area: int = 0, max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(min_area, max_area, silent)
        self.mean_area = mean_area
        self.std_area = std_area

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        image_height, image_width, _ = image.shape
        available_pixels = np.ones((image_height, image_width), dtype=bool)

        if mask is not None:
            available_pixels[np.logical_not(mask)] = False

        if self.silent:
            progress_bar = None
        else:
            progress_bar = tqdm.tqdm(total=np.count_nonzero(available_pixels))

        segments = []
        while np.any(available_pixels):
            patch_area = int(
                np.random.normal(self.mean_area, self.std_area, (1, )).clip(self.min_area, self.max_area)[0]
            )
            region = self.grow_region(available_pixels, patch_area)
            available_pixels[region] = False

            if progress_bar is not None:
                progress_bar.update(np.count_nonzero(region))

            segments.append(region)

        if progress_bar is not None:
            progress_bar.close()

        return segments

    def grow_region(self, binary_mask, target_area):
        coords = np.argwhere(binary_mask)

        start_y, start_x = coords[0]

        region = np.zeros_like(binary_mask, dtype=bool)
        region[start_y, start_x] = True

        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        boundary = [(start_y, start_x)]

        while np.sum(region) < target_area and len(boundary) > 0:
            idx = np.random.randint(0, len(boundary))
            y, x = boundary.pop(idx)

            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < binary_mask.shape[0] and 0 <= nx < binary_mask.shape[1]:
                    if binary_mask[ny, nx] and not region[ny, nx]:
                        region[ny, nx] = True
                        boundary.append((ny, nx))

        return region
