import logging

import bridson
import numpy as np
import scipy.ndimage as ndimage
import shapely
import tqdm
from scipy.spatial import Voronoi
import skimage.draw as sk_draw


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Function taken from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_background_gradient_field(
        image: np.ndarray, background_mask: np.ndarray, density: float = 50, erosion_iterations: int = 5,
        min_pixels: int = 100
):
    if erosion_iterations > 0:
        logging.info("Performing binary erosion to mitigate haloing...")
        for iterations in range(erosion_iterations, 0, -1):
            logging.info("Trying {} iterations...".format(iterations))

            eroded_background_mask = ndimage.binary_erosion(background_mask, iterations=iterations)

            if eroded_background_mask.sum() > min_pixels:
                background_mask = eroded_background_mask
                break

        else:
            logging.info(
                "The mask is too sparse to perform erosion! Haloing may appear in the background gradient field!"
            )

    size = background_mask.shape[::-1]

    logging.info("Synthesizing query points...")
    query_points = np.array(bridson.poisson_disc_samples(*size, density)).astype(int)

    logging.info("Constructing Voronoi diagram around query points...")
    vor = Voronoi(query_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    accepted_colors = []
    accepted_points = []

    logging.info("Processing each Voronoi region...")
    for region in tqdm.tqdm(regions):
        if -1 in region:
            continue

        if len(region) == 0:
            continue

        if region[0] != region[-1]:
            region.append(region[0])

        polygon_exterior = vertices[region]

        y = polygon_exterior[:, 1]
        x = polygon_exterior[:, 0]
        coordinates = sk_draw.polygon(x, y, image.shape[:2])

        color_mask = np.zeros_like(background_mask)
        color_mask[coordinates] = True
        color_mask[np.logical_not(background_mask)] = False

        if np.any(color_mask):
            accepted_colors.append(np.median(image[color_mask], axis=0))
            accepted_points.append(shapely.Polygon(polygon_exterior).centroid.coords[0])

    logging.info("Identified {} gradient data points".format(len(accepted_points)))
    return np.array(accepted_colors), np.array(accepted_points), density
