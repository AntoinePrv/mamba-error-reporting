import itertools
from typing import Iterable, TypeVar

T = TypeVar("T")


def pairwise(iterable: Iterable[T], last: T) -> Iterable[tuple[T, T]]:
    for a, b in itertools.pairwise(iterable):
        yield (a, b)
    yield (b, last)


def common_prefix(iterable: Iterable[str]) -> str:
    """Return the common prefix of a set of strings."""
    all_same = lambda chars: len(set(chars)) == 1
    return "".join(t[0] for t in itertools.takewhile(all_same, zip(*iterable)))


def repr_trunc(seq: list[str], sep: str = ", ", etc: str = "...", threshold: int = 5, show: tuple[int, int] = (2, 1)) -> str:
    if len(seq) < threshold:
        return sep.join(seq)
    show_head, show_tail = show
    return sep.join(seq[:show_head] + [etc] + seq[-show_tail:])
