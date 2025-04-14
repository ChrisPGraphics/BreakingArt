import typing

import numpy as np

import texton_categorization.base_categorization as base_categorization
import vector_node


class NearestColorCategorization(base_categorization.BaseCategorization):
    def __init__(self, colors: list):
        super().__init__()
        self.colors = np.array(colors) / 255

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        for polygon in polygons:
            polygon.category = np.argmin(np.linalg.norm(polygon.color - self.colors, axis=1)) + 1


class AreaPercentileCategorization(base_categorization.BaseCategorization):
    def __init__(self, percentiles: int):
        super().__init__()
        self.percentiles = percentiles

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        areas = [c.get_area() for c in polygons]

        quarts = np.linspace(0, 100, self.percentiles + 1)[1: -1]
        quart_range = np.percentile(areas, quarts)

        for i in range(len(polygons)):
            area = areas[i]

            for j, max_size in enumerate(quart_range):
                if area < max_size:
                    polygons[i].category = j + 1

                    break

            else:
                polygons[i].category = self.percentiles


class AreaBinaryThresholdCategorization(base_categorization.BaseCategorization):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        for polygon in polygons:
            if polygon.get_area() <= self.threshold:
                polygon.category = 1
            else:
                polygon.category = 2


class ConstantValueCategorization(base_categorization.BaseCategorization):
    def __init__(self, category: int = 1):
        super().__init__()
        self.category = category

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        for polygon in polygons:
            polygon.category = self.category


class UniqueColorCategorization(base_categorization.BaseCategorization):
    def __init__(self, tolerance: float = 0.01):
        super().__init__()
        self.tolerance = tolerance

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        known_colors = [polygons[0].color]
        for polygon in polygons:
            color_distances = np.linalg.norm(polygon.color - known_colors, axis=1)
            below_threshold = np.where(color_distances < self.tolerance)[0]

            if len(below_threshold) == 0:
                known_colors.append(polygon.color)
                polygon.category = len(known_colors)

            else:
                polygon.category = below_threshold[0] + 1

        return known_colors
