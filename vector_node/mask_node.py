import cv2
import numpy as np
import scipy.ndimage as ndimage
import shapely
import tqdm

import common
import vector_node.base_node as base_node
import vector_node.vector_node as vector_node


class MaskNode(base_node.BaseNode):
    def __init__(self, mask: np.ndarray, color: np.ndarray = None):
        super().__init__()
        self.mask = mask
        self.color = color

    @classmethod
    def from_image(cls, image: np.ndarray):
        return cls(np.ones(image.shape[:2], dtype=bool))

    def to_vector_node(self, pad: int = 5, silent: bool = True) -> vector_node.VectorNode:
        node = vector_node.VectorNode(self.to_polygon(pad=pad), color=self.color)

        iterator = self.children if silent else tqdm.tqdm(self.children)
        for child in iterator:
            node.add_child(child.to_vector_node(pad=pad, silent=True))

        return node

    def to_polygon(self, pad: int = 5) -> np.ndarray:
        mask = self.mask

        if pad > 0:
            mask = np.pad(self.mask, ((pad, pad), (pad, pad)), mode='constant')

        contours, _ = cv2.findContours(mask.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []

        for contour in contours:
            coords = []

            for point in contour:
                coords.append([point[0][1] - pad, point[0][0] - pad])

            if len(coords) < 3:
                continue

            polygons.append(shapely.Polygon(coords))

        polygon = polygons[np.argmax([p.area for p in polygons])]

        return np.array(polygon.exterior.coords)

    def get_area(self) -> int:
        return self.mask.sum()

    def fill_holes(self):
        return ndimage.binary_fill_holes(self.mask.astype(bool))

    def union(self, mask: np.ndarray):
        return np.logical_or(self.mask, mask)

    def intersection(self, mask: np.ndarray):
        return np.logical_and(self.mask, mask)

    def difference(self, mask: np.ndarray):
        return np.logical_and(self.mask, np.logical_not(mask))

    def dilate(self, iterations: int = 1, mask: np.ndarray = None):
        return common.binary_operations.dilate(self.mask, iterations=iterations, mask=mask)

    def is_touching(self, mask: np.ndarray):
        return np.any(np.logical_and(self.dilate(1), mask))
