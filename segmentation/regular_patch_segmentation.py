import logging
import math
import typing

import numpy as np
import shapely
import skimage.draw as sk_draw

import segmentation.base_segmentation as base_segmentation


class RegularPatchSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, polygon_boundary: typing.Union[np.ndarray, shapely.Polygon, list], silent: bool = False
    ):
        super().__init__(0, math.inf, silent)

        if isinstance(polygon_boundary, shapely.Polygon):
            polygon_boundary = polygon_boundary.boundary.coords

        patch_boundary = np.array(polygon_boundary)
        patch_boundary -= patch_boundary.min(axis=0)

        y = patch_boundary[:, 0]
        x = patch_boundary[:, 1]
        coordinates = sk_draw.polygon(x, y)

        mask_size = np.ceil(patch_boundary.max(axis=0)[::-1]).astype(int) + 1

        self.patch_mask = np.zeros(mask_size, dtype=bool)
        self.patch_mask[coordinates] = True

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        patch_height, patch_width = self.patch_mask.shape
        image_height, image_width, _ = image.shape

        patches_vertical = image_height // patch_height
        patches_horizontal = image_width // patch_width

        segmented = np.zeros([image_height, image_width], dtype=int)

        patch_number = 1
        for i in range(patches_vertical):
            for j in range(patches_horizontal):
                start_y = i * patch_height
                start_x = j * patch_width

                segmented[
                start_y:start_y + patch_height, start_x:start_x + patch_width
                ] = self.patch_mask * patch_number
                patch_number += 1

        logging.warning("Only {:.2f}% of the image could be covered by specified patch shape!".format(
            np.count_nonzero(segmented) / (image_height * image_width) * 100
        ))

        return self._masks_from_binary(segmented, mask)
