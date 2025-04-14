import logging

import bridson
import numpy as np
import tqdm

import vector_node


def generate_secondary_texton_distro(
        shape: tuple, distances: np.ndarray, secondary_textons: vector_node.VectorNode,
        background_color: np.ndarray = None, percentile: int = 40
):
    result = vector_node.VectorNode.from_rectangle(shape, color=background_color)

    if len(distances) == 0:
        logging.warning("No background textons found! Creating solid colored background...")
        return result

    radius = np.percentile(distances, percentile)

    logging.info("Building point distribution...")
    points = bridson.poisson_disc_samples(*shape, radius)

    choices = len(secondary_textons.children)

    logging.info("Placing polygons at points...")
    for point in tqdm.tqdm(points):
        polygon_index = np.random.randint(0, choices)
        texton: vector_node.VectorNode = secondary_textons.children[polygon_index].copy(deep_copy=False)
        texton.set_centroid(point)
        result.add_child(texton)

    logging.info("Sorting nodes by area...")
    result.children.sort(key=lambda x: x.get_area(), reverse=False)

    for child in result.children:
        child.set_centroid(child.get_centroid(yx=True))

    return result
