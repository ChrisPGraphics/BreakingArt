import abc
import collections
import typing

import common


class BaseNode(common.SavableObject, abc.ABC):
    def __init__(self):
        self.children: typing.List[typing.Self] = []

    def pre_order_traversal(self, include_self: bool = True) -> typing.Iterable[typing.Self]:
        if include_self:
            yield self

        for child in self.children:
            yield from child.pre_order_traversal()

    def post_order_traversal(self, include_self: bool = True) -> typing.Iterable[typing.Self]:
        for child in self.children:
            yield from child.post_order_traversal()

        if include_self:
            yield self

    def level_order_traversal(self, include_self: bool = True) -> typing.Iterable[typing.Self]:
        queue = collections.deque([self])

        while queue:
            node = queue.popleft()

            if include_self:
                yield node

            elif node != self:
                yield node

            queue.extend(node.children)

    def get_child_count(self) -> int:
        return len(self.children)

    def get_descendant_count(self, include_self: bool = True) -> int:
        count = 0
        for _ in self.level_order_traversal(include_self=include_self):
            count += 1

        return count

    def add_child(self, child: typing.Self):
        self.children.append(child)

    def add_children(self, children: typing.List[typing.Self]):
        self.children.extend(children)

    def clear_children(self):
        self.children.clear()

    def remove_child(self, child: typing.Self):
        self.children.remove(child)
