import itertools
from typing import Iterable, TypeVar

T = TypeVar("T")


def pairwise(iterable: Iterable[T], last: T) -> Iterable[tuple[T, T]]:
    for a, b in itertools.pairwise(iterable):
        yield (a, b)
    yield (b, last)
