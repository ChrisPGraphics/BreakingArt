import math
import typing

import PIL.Image
import PIL.ImageDraw
import cv2
import numpy as np
import shapely
import shapely.affinity
import shapely.coords
import svgwrite
import svgwrite.shapes
import tqdm
from matplotlib import pyplot as plt

import vector_node.base_node as base_node

if typing.TYPE_CHECKING:
    import analysis.descriptors as descriptors

POLYGON_TYPE = typing.Union[np.ndarray, 'VectorNode', shapely.Polygon]


class VectorNode(base_node.BaseNode):
    def __init__(
            self, exterior: typing.Union[np.ndarray, list], category: int = None,
            color: typing.Union[np.ndarray, tuple] = None
    ):
        super().__init__()
        if isinstance(exterior, (list, shapely.coords.CoordinateSequence)):
            exterior = np.array(exterior).astype(np.float32)

        self.exterior = exterior
        self.category = category
        self.color = color
        self.color_delta = None
        self.synthetic: bool = False
        self.descriptor: 'descriptors.Descriptor' = None
        self.transformations = []

        self._cached_shapely = None
        self._cached_shapely_exterior = None

    @classmethod
    def from_rectangle(cls, size: tuple, category: int = None, color: np.ndarray = None) -> 'VectorNode':
        size = size[::-1]
        return cls([(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1]), (0, 0)], category, color)

    def get_children_by_category(self) -> dict:
        result = {}

        for child in self.children:
            if child.category not in result:
                result[child.category] = []

            result[child.category].append(child)

        return result

    def as_shapely(self) -> shapely.Polygon:

        try:
            if self._cached_shapely is not None:
                if np.array_equal(self._cached_shapely_exterior, self.exterior):
                    return self._cached_shapely

        except AttributeError:
            pass

        p = shapely.Polygon(self.exterior)

        self._cached_shapely_exterior = self.exterior.copy()
        self._cached_shapely = p

        return p

    def from_shapely(self, polygon: shapely.Polygon):
        self.exterior = np.array(polygon.exterior.coords).astype(float)

    def get_centroid(self, yx: bool = False) -> np.ndarray:
        centroid = np.array(self.as_shapely().centroid.coords)[0]

        if yx:
            centroid = centroid[::-1]

        return centroid

    def set_centroid(self, centroid: typing.Union[np.ndarray, tuple]):
        self.move_centroid(centroid - self.get_centroid())

    def move_centroid(self, delta: typing.Union[np.ndarray, tuple]):
        self.exterior = self.exterior.astype(float) + delta

        for child in self.children:
            child.move_centroid(delta)

    def get_child_centroids(self) -> np.ndarray:
        return np.array([i.get_centroid() for i in self.children])

    def get_polygon_coordinate_pairs(self, yx: bool = False, normalized: bool = False) -> np.ndarray:
        coords = self.exterior.copy()

        if normalized:
            coords = self.exterior - self.get_centroid()

        if yx:
            coords = np.flip(coords, axis=1)

        return coords

    def get_polygon_split_coordinates(self, yx: bool = False, normalized: bool = False) -> np.ndarray:
        coords = self.get_polygon_coordinate_pairs(yx, normalized)
        return np.array([coords[:, 0], coords[:, 1]])

    def get_unique_child_categories(self):
        return np.unique([c.category for c in self.children])

    def get_area(self) -> float:
        return self.as_shapely().area

    def get_perimeter(self) -> float:
        return self.as_shapely().length

    def get_polsby_popper_compactness(self) -> float:
        return 4 * math.pi * (self.get_area() / self.get_perimeter() ** 2)

    def get_schwartzberg_compactness(self) -> float:
        return 1 / (self.get_perimeter() / (2 * math.pi * math.sqrt(self.get_area() / math.pi)))

    def get_length_width_ratio(self) -> float:
        return self.get_bounding_height() / self.get_bounding_width()

    def get_reock_score(self) -> float:
        return self.as_shapely().area / self.get_bounding_circle_area()

    def get_convex_hull_score(self) -> float:
        return self.as_shapely().area / self.get_convex_hull().area

    def get_elongation(self) -> float:
        area = self.get_area()
        perimeter = self.get_perimeter()

        return min(area, perimeter) / max(area, perimeter)

    def get_convex_hull(self) -> shapely.Polygon:
        return self.as_shapely().convex_hull

    def get_bounding_box(self) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds

        return np.array([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
        ])

    def get_bounding_height(self, as_int: bool = False) -> float:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds
        if as_int:
            return int(round(max_x - min_x))
        else:
            return max_x - min_x

    def get_bounding_width(self, as_int: bool = False) -> float:
        min_x, min_y, max_x, max_y = self.as_shapely().bounds
        if as_int:
            return int(round(max_y - min_y))
        else:
            return max_y - min_y

    def distance_to_point(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.get_centroid())

    @staticmethod
    def _convert_to_shapely(polygon) -> shapely.Polygon:
        if isinstance(polygon, VectorNode):
            polygon = polygon.as_shapely()

        elif not isinstance(polygon, shapely.Polygon):
            polygon = shapely.Polygon(polygon)

        return polygon

    def get_bounding_circle(self) -> shapely.Polygon:
        return shapely.minimum_bounding_circle(self.as_shapely())

    def get_bounding_circle_radius(self) -> float:
        return shapely.minimum_bounding_radius(self.as_shapely())

    def get_bounding_circle_area(self) -> float:
        return math.pi * self.get_bounding_circle_radius() ** 2

    def distance_to_polygon(self, polygon: POLYGON_TYPE) -> float:
        polygon = self._convert_to_shapely(polygon)
        return self.distance_to_point(np.array(polygon.centroid.coords)[0])

    def angle_to_point(self, point: np.ndarray) -> float:
        return math.atan2(*(point - self.get_centroid())[::-1])

    def angle_to_polygon(self, polygon: POLYGON_TYPE) -> float:
        polygon = self._convert_to_shapely(polygon)
        return self.angle_to_point(np.array(polygon.centroid.coords)[0])

    def get_overlap_percent(self, polygon: POLYGON_TYPE) -> float:
        polygon = self._convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        return polygon.intersection(self_polygon).area / min(polygon.area, self_polygon.area)

    def is_fully_contained(self, polygon: POLYGON_TYPE) -> bool:
        polygon = self._convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        return polygon.intersection(self_polygon).area == polygon.area

    def is_touching(self, polygon: POLYGON_TYPE, border_touching: bool = True) -> bool:
        polygon = self._convert_to_shapely(polygon)
        self_polygon = self.as_shapely()

        if border_touching:
            return polygon.intersects(self_polygon)

        return polygon.intersection(self_polygon).area > 0

    def contains_point(self, point) -> bool:
        return self.as_shapely().contains(shapely.Point(point))

    def refit_to_parent(self, recursive: bool = True):
        s = self.as_shapely()

        for child in self.children[:]:
            try:
                polygon = s.intersection(child.as_shapely())
            except shapely.GEOSException:
                polygon = s.buffer(0).intersection(child.as_shapely().buffer(0))

            if isinstance(
                    polygon, (shapely.LineString, shapely.Point, shapely.MultiPoint, shapely.MultiLineString)
            ) or polygon.is_empty:

                self.children.remove(child)

            elif isinstance(polygon, (shapely.MultiPolygon, shapely.GeometryCollection)):
                self.children.remove(child)

                for p in polygon.geoms:
                    if isinstance(p, shapely.Polygon):
                        node = child.copy()
                        node.from_shapely(p)
                        self.add_child(node)

                        if recursive:
                            node.refit_to_parent()

            else:
                child.from_shapely(polygon)

                if recursive:
                    child.refit_to_parent()

    def to_svg(self, filename: str):
        dwg = svgwrite.Drawing(filename, profile='tiny', size=(self.get_bounding_width(), self.get_bounding_height()))

        for node in self.level_order_traversal():
            if node.color is None:
                continue

            dwg.add(
                svgwrite.shapes.Polygon(
                    node.get_polygon_coordinate_pairs(yx=True).tolist(),
                    fill=svgwrite.rgb(*node.color * 100, '%')
                )
            )

        dwg.save()

    def color_to_int(self, scale: float = 255) -> np.ndarray:
        return (np.array(self.color) * scale).astype(int)

    def to_raster(
            self, filename: str, background_image: np.ndarray = None, background_color: np.ndarray = None,
            transparent_background: bool = True
    ):
        if background_color is None:
            background_color = (0, 0, 0) if self.color is None else self.color_to_int()

        else:
            background_color = (background_color * 255).astype(np.uint8)

        if len(background_color) == 3 and transparent_background:
            background_color = list(background_color)
            background_color.append(0)

        surface = PIL.Image.new(
            "RGBA" if transparent_background else "RGB",
            (int(self.get_bounding_width()), int(self.get_bounding_height())),
            color=tuple(background_color)
        )

        if background_image is not None:
            surface.paste(PIL.Image.fromarray(np.uint8(background_image * 255)))

        draw = PIL.ImageDraw.Draw(surface, 'RGBA')

        for child in self.pre_order_traversal(include_self=False):
            fill_color = child.color_to_int()

            if fill_color is None:
                continue

            fill_color = list(fill_color)
            if len(fill_color) == 3:
                fill_color.append(255)

            coordinates = [tuple(c) for c in child.get_polygon_coordinate_pairs(yx=True).astype(int)]
            if len(coordinates) < 2:
                continue

            draw.polygon(coordinates, fill=tuple(fill_color))

        surface.save(filename)

    def rotate(self, angle: float):
        centroid = tuple(self.get_centroid())

        for child in self.level_order_traversal():
            rotated = shapely.affinity.rotate(child.as_shapely(), angle, centroid, use_radians=True)
            child.from_shapely(rotated)

    def scale(self, x_scale: float = 1, y_scale: float = 1):
        centroid = tuple(self.get_centroid())

        for child in self.level_order_traversal():
            scaled = shapely.affinity.scale(child.as_shapely(), x_scale, y_scale, origin=centroid)
            child.from_shapely(scaled)

    def synthesize_children(
            self, quantity: float, top: float = 0.25,
            rotation_range: tuple = (-math.pi / 4, math.pi / 4),
            scale_range: tuple = (8/9, 9/8)
    ):
        if scale_range is None and rotation_range is None:
            return

        total_candidates = int(len(self.children) * top)

        if quantity < 1:
            quantity = len(self.children) * quantity

        quantity = int(quantity)

        areas = np.array([c.get_area() for c in self.children])
        compactness = np.array([c.get_polsby_popper_compactness() for c in self.children])

        areas -= areas.min()
        areas /= areas.max()

        compactness -= compactness.min()
        compactness /= compactness.max()

        coordinates = np.column_stack((areas, compactness))
        distances = np.linalg.norm(coordinates - [1, 1], axis=1)

        candidate_indices = np.argpartition(distances, total_candidates)[-total_candidates:]

        children = []
        for _ in tqdm.tqdm(range(quantity), total=quantity):
            parent_index = np.random.choice(candidate_indices)
            new_child = self.children[parent_index].copy()

            if scale_range is not None:
                scale_x = np.random.uniform(*scale_range)
                scale_y = np.random.uniform(*scale_range)
                new_child.scale(scale_x, scale_y)

            if rotation_range is not None:
                angle = np.random.uniform(*rotation_range)
                new_child.rotate(angle)

            new_child.synthetic = True
            self.add_child(new_child)
            children.append(new_child)

        return children

    def get_category_density(self, normalize: bool = True) -> tuple:
        categories = [c.category for c in self.children if not c.synthetic]
        unique, counts = np.unique(categories, return_counts=True)

        if normalize:
            counts = counts.astype(np.float32) / len(self.children)

        return unique, counts

    def get_raster_coords(
            self, centroid: typing.Union[np.ndarray, tuple] = None, x_lim=None, y_lim=None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:

        coords = self.exterior.copy()

        if centroid is not None:
            coords += centroid - self.get_centroid()

        lower = coords.min(axis=0)
        upper = coords.max(axis=0)
        size = upper - lower

        mask = np.zeros((size + 1).astype(int), dtype=np.uint8)
        offset_coords = (coords - lower).astype(int)

        cv2.fillPoly(mask, [offset_coords[:, [1, 0]]], 1)
        coordinates = np.where(mask == 1)

        coordinates = np.array([(coordinates[0] + lower[0]).astype(int), (coordinates[1] + lower[1]).astype(int)])

        if x_lim is not None:
            mask = (x_lim[0] < coordinates[1]) & (coordinates[1] < x_lim[1]) & \
                   (y_lim[0] < coordinates[0]) & (coordinates[0] < y_lim[1])

            if not np.any(mask):
                raise OverflowError("Out of range")

            coordinates = (coordinates[0][mask], coordinates[1][mask])

        return coordinates

    def binary_rasterize(
            self, mask: np.ndarray, centroid: typing.Union[np.ndarray, tuple] = None,
            color: typing.Union[int, None] = 1, transpose: bool = True
    ):
        if color is None:
            color = self.category

        if centroid is None:
            coords = self.exterior

        else:
            coords = self.exterior + (centroid - self.get_centroid())

        points = [coords.astype(int)[:, [1, 0]]] if transpose else [coords.astype(int)]

        if mask.dtype == np.uint8:
            cv2.fillPoly(mask, points, int(color))

        else:
            write_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillPoly(write_mask, points, 1)
            mask[write_mask == 1] = int(color)

    def flat_plot(self, color: typing.Union[np.ndarray, tuple] = None, fill: bool = True, line_width: float = None):
        if color is None:
            color = self.color

        if fill:
            plt.fill(*self.get_polygon_split_coordinates(yx=True), c=color)

        else:
            plt.plot(*self.get_polygon_split_coordinates(yx=True), c=color, linewidth=line_width)

    def plot(
            self, surface_only: bool = False, include_self: bool = True, fill: bool = True
    ):
        generator = [self] + self.children if surface_only else self.pre_order_traversal()
        for child in generator:
            if include_self:
                child.flat_plot(fill=fill)

            elif child != self:
                child.flat_plot(fill=fill)

    def intersection_over_union(self, other: 'VectorNode') -> float:
        self_shapely = self.as_shapely()
        other_shapely = other.as_shapely()

        self_centroid = self_shapely.centroid
        other_centroid = other_shapely.centroid

        dx = self_centroid.x - other_centroid.x
        dy = self_centroid.y - other_centroid.y

        other_shapely = shapely.affinity.translate(other_shapely, xoff=dx, yoff=dy)

        return shapely.intersection(self_shapely, other_shapely).area / shapely.union(self_shapely, other_shapely).area
