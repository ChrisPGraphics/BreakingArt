import json
import logging
import math
import os.path
import shutil
import typing

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
import shapely

import common
import vector_node


class Weights(common.SavableObject):
    def __init__(
            self,
            empty_area_weight: float = -0.2,
            missed_area_weight: float = -0.4,
            mismatched_area_weight: float = -0.5,
            target_area_weight: float = 0.5,
            same_overlap_area_weight: float = -0.5,
            different_overlap_area_weight: float = -0.5,
    ):
        self.empty_area_weight = empty_area_weight
        self.missed_area_weight = missed_area_weight
        self.mismatched_area_weight = mismatched_area_weight
        self.target_area_weight = target_area_weight
        self.same_overlap_area_weight = same_overlap_area_weight
        self.different_overlap_area_weight = different_overlap_area_weight

    def to_array(self) -> np.ndarray:
        return np.array([
            self.empty_area_weight,
            self.missed_area_weight,
            self.mismatched_area_weight,
            self.target_area_weight,
            self.same_overlap_area_weight,
            self.different_overlap_area_weight
        ])

    @classmethod
    def from_array(cls, array) -> 'typing.Self':
        return cls(*array)

    def to_json(self, filename, metadata: dict):
        result = {
            "weights": self.__dict__,
            "metadata": metadata
        }

        with open(filename, 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4))

    @staticmethod
    def from_json(filename: str) -> typing.Tuple['Weights', dict]:
        with open(filename, 'r') as f:
            data = json.loads(f.read())

        weights = Weights(**data["weights"])
        metadata = data["metadata"]

        return weights, metadata


def get_probability_distribution(mask):
    distro = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
    return distro / distro.sum()


def sample_probability_distribution(distro: np.ndarray, count: int = None) -> typing.Tuple[int, int]:
    p_flat = distro.ravel()
    ind = np.arange(len(p_flat))

    # random_index = np.random.choice(ind, size=count, p=p_flat)
    random_index = common.generator.sample_distribution(ind, p_flat, count)

    rows, cols = distro.shape
    row = random_index // cols
    col = random_index % cols

    return row, col


@numba.jit(nopython=True)
def get_connected_component(binary_array, start_row, start_col):
    rows, cols = binary_array.shape
    if binary_array[start_row, start_col] == 0:
        return np.zeros_like(binary_array, dtype=np.uint8)

    mask = np.zeros_like(binary_array, dtype=np.uint8)
    stack = [(start_row, start_col)]
    mask[start_row, start_col] = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        row, col = stack.pop()

        for dr, dc in directions:
            r, c = row + dr, col + dc

            if rows > r >= 0 == mask[r, c] and 0 <= c < cols and binary_array[r, c] == 1:
                mask[r, c] = 1
                stack.append((r, c))

    return mask


def count_connected_pixels(mask: np.ndarray, coords):
    polygon_mask = np.zeros_like(mask)
    polygon_mask[coords] = True

    result = np.zeros_like(mask, dtype=bool)
    intersection = np.logical_and(mask, polygon_mask)

    nonzero_indices = np.flatnonzero(intersection)

    while len(nonzero_indices) > 0:
        seed_index = nonzero_indices[0]
        rows, cols = intersection.shape
        row = seed_index // cols
        col = seed_index % cols

        connected_component = get_connected_component(mask, row, col)
        result = np.logical_or(result, connected_component)

        intersection = np.logical_and(intersection, np.logical_not(connected_component))
        nonzero_indices = np.flatnonzero(intersection)

    return common.binary_operations.count_true(result)


def count_connected_pixels2(mask: np.ndarray, coords):
    original_mask = mask
    mask = mask.astype(int)
    mask[coords] = True

    cv2.floodFill(mask, None, (coords[1][0], coords[0][0]), 255, 0, 0, flags=4)
    mask = mask == 255

    return common.binary_operations.count_true(np.logical_and(mask, original_mask))


def count_connected_pixels3(mask: np.ndarray, coords):
    # this is faster than temp_mask = mask.astype(np.uint8)
    temp_mask = np.zeros_like(mask, dtype=np.uint8)
    temp_mask[mask] = 1

    temp_mask[coords] = 1

    num, temp_mask, _, _ = cv2.floodFill(temp_mask, None, (coords[1][0], coords[0][0]), 255, flags=4)

    # return common.binary_operations.count_true(np.logical_and(mask, temp_mask == 255))
    return np.count_nonzero(np.logical_and(mask, temp_mask == 255))


def generate_primary_texton_distro(
        polygons: vector_node.VectorNode, size: typing.Tuple[int, int],
        placement_tries: int = 10, improvement_steps: int = 5, max_fails: int = 30,
        selection_probability_decay: float = 2, weights: Weights = Weights(), log_steps_directory: str = None,
        placed_descriptors: np.ndarray = None
) -> vector_node.VectorNode:

    def place_polygon(polygon_index: int, centroid: tuple):
        placed_texton = textons[polygon_index].copy()
        placed_texton.set_centroid(centroid)

        descriptor_center = placed_texton.descriptor.center

        top_left = int(centroid[0] - descriptor_center[0]), int(centroid[1] - descriptor_center[1])

        descriptor_shape = placed_texton.descriptor.descriptor.shape
        placed_descriptor = placed_texton.descriptor.descriptor

        y_range = np.arange(descriptor_shape[0]) + top_left[1]
        x_range = np.arange(descriptor_shape[1]) + top_left[0]

        valid_y = (y_range >= 0) & (y_range < size[1])
        valid_x = (x_range >= 0) & (x_range < size[0])

        y_grid, x_grid = np.meshgrid(np.where(valid_y)[0], np.where(valid_x)[0], indexing='ij')

        offset_y = y_grid + top_left[1]
        offset_x = x_grid + top_left[0]

        valid_positions = (placed_descriptor[y_grid, x_grid] != 0) & \
                          (placed_descriptors[offset_y, offset_x] == 0) & \
                          (placed_polygons[offset_y, offset_x] == 0)

        placed_descriptors[offset_y[valid_positions], offset_x[valid_positions]] = placed_descriptor[
            y_grid[valid_positions], x_grid[valid_positions]
        ]

        placed_texton.binary_rasterize(placed_descriptors, color=-1, transpose=False)
        placed_texton.binary_rasterize(placed_polygons, color=placed_texton.category, transpose=False)

        result.add_child(placed_texton)

    def score_placement(polygon_index: int, centroid: tuple):
        placed_texton = textons[polygon_index]
        try:
            coords = placed_texton.get_raster_coords(centroid, x_lim=(0, size[0]), y_lim=(0, size[1]))
        except OverflowError:
            return -math.inf

        descriptor_pixels = placed_descriptors[coords]
        polygon_pixels = placed_polygons[coords]

        target_area = np.sum(descriptor_pixels == placed_texton.category)
        mismatched_area = np.sum((descriptor_pixels != placed_texton.category) & (descriptor_pixels > 0))

        if weights.missed_area_weight == 0:
            missed_area = 0

        else:
            missed_mask = placed_descriptors == placed_texton.category
            if initial_placed_descriptors is not None:
                missed_mask = np.logical_and(missed_mask, np.logical_not(initial_placed_descriptors))

            missed_area = count_connected_pixels3(missed_mask, coords)
            missed_area -= target_area

        empty_area = np.sum(descriptor_pixels <= 0)

        same_overlap_area = np.sum(polygon_pixels == placed_texton.category)
        different_overlap_area = np.sum((polygon_pixels != placed_texton.category) & (polygon_pixels != 0))

        return (
                weights.empty_area_weight * empty_area +
                weights.missed_area_weight * missed_area +
                weights.mismatched_area_weight * mismatched_area +
                weights.target_area_weight * target_area +
                weights.same_overlap_area_weight * same_overlap_area +
                weights.different_overlap_area_weight * different_overlap_area
        )

    if log_steps_directory is not None:
        if os.path.isdir(log_steps_directory):
            shutil.rmtree(log_steps_directory)

        os.makedirs(os.path.join(log_steps_directory, "polygons"))
        os.makedirs(os.path.join(log_steps_directory, "descriptors"))

    size = (int(size[0]), int(size[1]))
    result = vector_node.VectorNode.from_rectangle(size, None, polygons.color)

    placed_polygons = np.zeros(size[::-1], dtype=int)

    if placed_descriptors is None:
        initial_placed_descriptors = None
        placed_descriptors = np.zeros(size[::-1], dtype=int)

    else:
        initial_placed_descriptors = placed_descriptors.copy()

    textons = polygons.children

    texton_choices = len(textons)
    initial_index = np.random.randint(0, texton_choices)
    place_polygon(initial_index, (size[0] / 2, size[1] / 2))

    unique_categories = list(np.unique([i.category for i in textons]))
    categorized_texton_indices = [
        [textons.index(texton) for texton in textons if texton.category == category] for category in unique_categories
    ]
    texton_selection_probabilities = [np.ones(len(d)) / len(d) for d in categorized_texton_indices]

    iteration = 0
    fails = 0

    while True:
        iteration += 1
        probability_distribution = get_probability_distribution(placed_descriptors > 0)

        best_score = -math.inf
        best_centroid = None
        best_texton_index = None

        center_pixels = sample_probability_distribution(probability_distribution, count=placement_tries)
        for center_pixel in zip(*center_pixels):
            category_color = placed_descriptors[*center_pixel]

            try:
                category_index = unique_categories.index(category_color)
            except ValueError:

                if category_color != 0:
                    logging.warning(
                        "No polygons in the inclusion zone are of category {0}! "
                        "Clearing all category {0} pixels".format(
                            category_color
                        )
                    )

                placed_descriptors[placed_descriptors == category_color] = 0
                continue

            texton_index = categorized_texton_indices[category_index][common.generator.sample_distribution_once(
                texton_selection_probabilities[category_index]
            )]

            score = score_placement(texton_index, center_pixel)

            if score >= best_score:
                best_score = score
                best_centroid = center_pixel
                best_texton_index = texton_index

        if best_centroid is None:
            logging.warning("No proposed pixels are left! Terminating early.")
            break

        checked_centroids = [best_centroid]
        for _ in range(improvement_steps):
            checked = 0

            for new_center_pixel in [
                tuple(best_centroid + np.array([0, 1])),
                tuple(best_centroid + np.array([0, -1])),
                tuple(best_centroid + np.array([1, 0])),
                tuple(best_centroid + np.array([-1, 0])),
            ]:
                if new_center_pixel in checked_centroids:
                    continue

                checked += 1
                checked_centroids.append(new_center_pixel)

                try:
                    new_score = score_placement(best_texton_index, new_center_pixel)
                except OverflowError:
                    continue

                if new_score >= best_score:
                    best_score = new_score
                    best_centroid = new_center_pixel

            if checked == 0:
                break

        placed_descriptors[
            textons[best_texton_index].get_raster_coords(best_centroid, x_lim=(0, size[0]), y_lim=(0, size[1]))
        ] = -1

        if best_score < 0:
            fails += 1

            if fails > max_fails:
                break

            logging.info(
                "Iteration: {:>4}, Best Score: {:>9.3f}, Consecutive Fails: {:>2}".format(
                    iteration, best_score, fails
                )
            )

        else:
            fails = 0
            place_polygon(best_texton_index, best_centroid[::-1])

            logging.info(
                "Iteration: {:>4}, Best Score: {:>9.3f}, Placed Polygons: {:>4}".format(
                    iteration, best_score, len(result.children)
                ))

            category_index = unique_categories.index(textons[best_texton_index].category)
            texton_selection_probabilities[category_index] /= selection_probability_decay
            total_probability = texton_selection_probabilities[category_index].sum()
            texton_selection_probabilities[category_index] /= total_probability

        if log_steps_directory is not None:
            adjusted = (placed_polygons / np.max(unique_categories) * 255).astype(np.uint8)
            PIL.Image.fromarray(adjusted).save(
                os.path.join(log_steps_directory, "polygons", "{:0>4}.png".format(iteration))
            )

            adjusted = (placed_descriptors / (np.max(unique_categories) + 1) * 255).astype(np.uint8)
            PIL.Image.fromarray(adjusted).save(
                os.path.join(log_steps_directory, "descriptors", "{:0>4}.png".format(iteration))
            )

    # result.children.sort(key=lambda x: x.get_area(), reverse=True)

    for child in result.children:
        child.set_centroid(child.get_centroid(yx=True))

    return result


def get_coverage(distribution: vector_node.VectorNode, inset: int = 75, debug: bool = False):
    union = shapely.unary_union(
        [texton.as_shapely().buffer(0) for texton in distribution.children]
    )

    min_x, min_y, max_x, max_y = union.bounds

    bounding_polygon = shapely.Polygon([
        [min_y, min_x],
        [min_y, max_x],
        [max_y, max_x],
        [max_y, min_x],
    ])

    bounding_polygon = bounding_polygon.buffer(-inset)
    union = shapely.intersection(union, bounding_polygon)

    if debug:
        distribution.plot(surface_only=True, include_self=False)
        plt.plot(*bounding_polygon.exterior.xy)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    try:
        return union.area / bounding_polygon.area

    except ZeroDivisionError:
        return 10


def get_category_area(distribution: vector_node.VectorNode):
    # total_area = distribution.get_area()

    categories = np.unique([i.category for i in distribution.children])
    coverage_area = [
        shapely.unary_union(
            [texton.as_shapely().buffer(0) for texton in distribution.children if texton.category == category]
        )
        for category in categories
    ]

    min_x, min_y, max_x, max_y = shapely.unary_union(coverage_area).bounds
    total_area = (max_x - min_x) * (max_y - min_y)

    return categories, [polygon.area / total_area for polygon in coverage_area]


def primary_density_cleanup(
        exemplar: vector_node.VectorNode, result: vector_node.VectorNode, skip_categories: typing.List[int] = None
):
    total_area = result.get_area()

    if skip_categories is None:
        skip_categories = []

    categories, target_coverage_area = get_category_area(exemplar)
    _, current_coverage_area = get_category_area(result)

    for category, target_area, current_area in zip(categories, target_coverage_area, current_coverage_area):
        if current_area <= target_area:
            continue

        if category in skip_categories:
            continue

        removal_candidates = [texton for texton in result.children if texton.category == category]

        while True:
            try:
                removal_index = np.random.randint(len(removal_candidates))
            except ValueError:
                break

            removed_polygon = removal_candidates[removal_index]

            current_area -= removed_polygon.get_area() / total_area
            removal_candidates.remove(removed_polygon)

            if current_area <= target_area:
                if np.random.random() > 0.5:
                    result.remove_child(removed_polygon)

                break

            result.remove_child(removed_polygon)
