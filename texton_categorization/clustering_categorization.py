import abc
import logging
import typing

import sklearn.cluster as cluster

import texton_categorization.base_categorization as base_categorization
import numpy as np
import vector_node


class ClusteringCategorization(base_categorization.BaseCategorization, abc.ABC):
    def __init__(self, cluster_count: int = 15):
        super().__init__()
        self.cluster_count = cluster_count

    @abc.abstractmethod
    def get_polygon_properties(self, polygons: typing.List[vector_node.VectorNode]) -> list:
        pass

    def get_property_weights(self) -> list:
        pass

    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        cluster_count = self.cluster_count
        if len(polygons) == 1:
            logging.warning("{} clusters requested, but only 1 polygon is provided! Setting category to 1!")
            polygons[0].category = 1

            return

        elif len(polygons) < self.cluster_count:
            cluster_count = max(int(len(polygons) // 2), 2)
            logging.warning(
                "{} clusters requested, but only {} polygons are provided! "
                "Reducing number of clusters to {} to compensate.".format(
                    self.cluster_count, len(polygons), cluster_count
                )
            )

        categorized = self.get_polygon_properties(polygons)
        weights = self.get_property_weights()

        if weights is None:
            weights = np.ones(len(categorized[0]))

        categorized = np.array(categorized)
        maximum = categorized.max(axis=0)
        minimum = categorized.min(axis=0)

        denominator = maximum - minimum
        denominator[denominator == 0] = 1

        normalized = (categorized - minimum) / denominator
        weighted = normalized * weights

        model = cluster.KMeans(n_clusters=cluster_count, n_init=10, random_state=0)
        categories = model.fit(weighted).labels_

        for polygon, category in zip(polygons, categories):
            polygon.category = category + 1


class ColorCategorization(ClusteringCategorization):
    def get_polygon_properties(self, polygons: typing.List[vector_node.VectorNode]) -> list:
        return [p.color for p in polygons]


class ColorAreaCompactnessCategorization(ClusteringCategorization):
    def get_polygon_properties(self, polygons: typing.List[vector_node.VectorNode]) -> list:
        return [(*p.color, p.get_area(), p.get_polsby_popper_compactness()) for p in polygons]

    def get_property_weights(self) -> list:
        return [2, 2, 2, 1, 1]
