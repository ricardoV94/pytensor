from collections.abc import Iterable, Iterator, MutableSet
from typing import Any, Literal


class OrderedSet(MutableSet):
    """Ordered-set like data structure.

    Uses a dictionary with None values to store the set elements.
    Relying on the fact that Python dicts maintain insertion order.
    """

    __slots__ = ("values",)
    values: dict[Any, Literal[None]]

    def __init__(self, iterable: Iterable | None = None) -> None:
        if iterable is None:
            self.values = {}
        else:
            self.values = {value: None for value in iterable}

    def __contains__(self, value) -> bool:
        return value in self.values

    def __iter__(self) -> Iterator:
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)

    def add(self, value) -> None:
        self.values[value] = None

    def discard(self, value) -> None:
        try:
            del self.values[value]
        except KeyError:
            pass

    def copy(self) -> "OrderedSet":
        new_set = OrderedSet()
        new_set.values = self.values.copy()
        return new_set

    def update(self, other: Iterable) -> None:
        self.values.update({o: None for o in other})

    def union(self, other: Iterable) -> "OrderedSet":
        new_set = OrderedSet()
        new_set.values = self.values.copy() | {o: None for o in other}
        return new_set

    def difference_update(self, other: Iterable) -> None:
        self_values = self.values
        for value in other:
            try:
                del self_values[value]
            except KeyError:
                pass
