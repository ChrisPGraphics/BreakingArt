import logging
import typing

import numpy as np
import scipy.spatial as spatial
import shapely
import tqdm

import vector_node


def get_colinear_inter_element_lines(points: np.ndarray) -> list:
    p1, p2 = points[0], points[1]
    line_vector = p2 - p1

    line_unit_vector = line_vector / np.linalg.norm(line_vector)

    projections = np.dot(points - p1, line_unit_vector)

    sorted_indices = np.argsort(projections)
    sorted_points = points[sorted_indices]

    neighbors = [(sorted_points[i], sorted_points[i + 1]) for i in range(len(sorted_points) - 1)]

    return neighbors


def get_inter_element_lines(centroids: np.ndarray, buffer: int = 10) -> np.ndarray:
    min_x, min_y = np.min(centroids, axis=0)
    max_x, max_y = np.max(centroids, axis=0)

    if np.all(np.isclose(centroids[:, 0], centroids[0, 0])) or np.all(np.isclose(centroids[:, 1], centroids[0, 1])):
        logging.warning(
            "All points are colinear! Sorting points along line and using neighbouring points as triangulation"
        )
        triangulation_lines = get_colinear_inter_element_lines(centroids)

    else:
        triangulation = spatial.Delaunay(centroids)

        triangulation_lines = []
        for points in centroids[triangulation.simplices]:
            triangulation_lines.append([points[0], points[1]])
            triangulation_lines.append([points[1], points[2]])
            triangulation_lines.append([points[2], points[0]])

    if len(triangulation_lines) > 5:
        accepted_lines = []
        for line in triangulation_lines:
            start, end = line

            if start[0] <= min_x + buffer or start[0] >= max_x - buffer:
                continue

            if end[0] <= min_x + buffer or end[0] >= max_x - buffer:
                continue

            if start[1] <= min_y + buffer or start[1] >= max_y - buffer:
                continue

            if end[1] <= min_y + buffer or end[1] >= max_y - buffer:
                continue

            accepted_lines.append([start, end])

    else:
        accepted_lines = triangulation_lines

    return np.array(accepted_lines)


def get_secondary_texton_triangulation(
        primary_textons: vector_node.VectorNode, secondary_textons: vector_node.VectorNode, buffer: int = 10
) -> np.ndarray:

    centroids = secondary_textons.get_child_centroids()
    triangulation_lines = get_inter_element_lines(centroids, buffer)

    occlusions = [polygon.as_shapely().buffer(0) for polygon in primary_textons.children]

    accepted_lines = []

    logging.info("Removing occluded triangulation lines...")
    for start, end in tqdm.tqdm(triangulation_lines):
        polyline = shapely.LineString([start, end])

        for polygon in occlusions:
            if polygon.intersects(polyline):
                break

        else:
            accepted_lines.append([start, end])

    return np.array(accepted_lines)


def get_line_distances(lines: np.ndarray) -> np.ndarray:
    result = []

    for start, end in lines:
        result.append(np.linalg.norm(end - start))

    return np.array(result)
