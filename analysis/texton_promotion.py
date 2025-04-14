import logging
import typing

import numpy as np

import vector_node


def promote_textons(
        primary: vector_node.VectorNode, secondary: vector_node.VectorNode, size_multiplier: float = 1
) -> typing.List[vector_node.VectorNode]:

    area_median = np.median([i.get_area() for i in primary.children]) * size_multiplier

    promote = []
    for polygon in secondary.children:          # type: vector_node.VectorNode
        if polygon.get_area() >= area_median:
            promote.append(polygon)

    logging.info(
        "Promoting {} of {} background polygons ({:.3f}%)".format(
            len(promote), len(secondary.children), len(promote) / len(secondary.children) * 100
        )
    )
    for polygon in promote:
        primary.add_child(polygon)
        secondary.remove_child(polygon)

    return promote
