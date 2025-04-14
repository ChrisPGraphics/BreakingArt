import numpy as np

import common
import vector_node


class PrimaryTextonResult(common.SavableObject):
    def __init__(self, primary_textons: vector_node.VectorNode, descriptor_size: int):
        self.primary_textons = primary_textons
        self.descriptor_size = descriptor_size


class SecondaryTextonResult(common.SavableObject):
    def __init__(self, secondary_textons: vector_node.VectorNode, element_spacing: np.ndarray):
        self.secondary_textons = secondary_textons
        self.element_spacing = element_spacing


class GradientFieldResult(common.SavableObject):
    def __init__(self, points: np.ndarray, colors: np.ndarray, density: float):
        self.points = points
        self.colors = colors
        self.density = density
