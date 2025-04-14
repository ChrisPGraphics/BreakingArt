import logging
import typing

import numpy as np
import tqdm

import analysis.secondary_spacing as secondary_spacing
import vector_node

ACTIVE_AREA_DESCRIPTOR_WARNING = 4


class Descriptor:
    def __init__(self, descriptor: np.ndarray, center: typing.Tuple[int, int]):
        self.descriptor = descriptor
        self.center = center


def get_descriptors(textons: vector_node.VectorNode, image_size, average_included: float = 2.25) -> int:
    descriptor_size = estimate_descriptor_size(textons, average_included)
    logging.info("Estimated descriptor size is {0}x{0}".format(descriptor_size))

    active_width = image_size[1] - descriptor_size
    active_height = image_size[0] - descriptor_size

    if (
            active_width / descriptor_size < ACTIVE_AREA_DESCRIPTOR_WARNING or
            active_height / descriptor_size < ACTIVE_AREA_DESCRIPTOR_WARNING
    ):
        logging.critical(
            "The active area is less than {} times the descriptor size! "
            "Please provide a larger exemplar or a smaller descriptor size".format(ACTIVE_AREA_DESCRIPTOR_WARNING)
        )

    accepted_textons = []

    logging.info("Extracting neighbourhood of each texton...")
    for texton in tqdm.tqdm(textons.children):
        centroid = texton.get_centroid()
        min_x, min_y = centroid - descriptor_size
        max_x, max_y = centroid + descriptor_size

        if centroid[0] < descriptor_size or centroid[0] > image_size[1] - descriptor_size:
            continue

        if centroid[1] < descriptor_size or centroid[1] > image_size[0] - descriptor_size:
            continue

        accepted_textons.append(texton)
        included_textons = []
        descriptor = np.zeros(image_size[::-1], dtype=int)
        descriptor[int(min_x): int(max_x) + 1, int(min_y): int(max_y) + 1] = -1

        for other_texton in textons.children:
            if texton == other_texton:
                continue

            x = other_texton.exterior[:, 0]
            y = other_texton.exterior[:, 1]

            if np.any((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)):
                included_textons.append(other_texton)
                other_texton.binary_rasterize(descriptor, color=None)

        non_zero_indices = np.argwhere(descriptor != 0)

        row_min, col_min = non_zero_indices.min(axis=0)
        row_max, col_max = non_zero_indices.max(axis=0)

        descriptor = descriptor[row_min:row_max + 1, col_min:col_max + 1]

        center_pixel = int(round(centroid[0] - row_min)), int(round(centroid[1] - col_min))
        texton.descriptor = Descriptor(descriptor, center_pixel)

    textons.children = accepted_textons
    return descriptor_size


def estimate_descriptor_size(textons: vector_node.VectorNode, average_included: float = 2, buffer: int = 10) -> int:
    if average_included < 1:
        logging.warning("Average included is less than 1. This is too low to get plausible results.")

    elif average_included < 2:
        logging.warning("Average included is less than 2. Please use a value of 2 or more for best results.")

    centroids = textons.get_child_centroids()
    triangulation = secondary_spacing.get_inter_element_lines(centroids, buffer=buffer)
    element_distances = secondary_spacing.get_line_distances(triangulation)

    return int(np.ceil(np.mean(element_distances) * average_included))
