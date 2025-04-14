import abc
import typing

import vector_node


class BaseCategorization(abc.ABC):
    @abc.abstractmethod
    def categorize(self, polygons: typing.List[vector_node.VectorNode]):
        pass

    def get_algorithm_name(self) -> str:
        return self.__class__.__name__
