from typing import Literal

type Kind = Literal["word", "whitespace", "punctuation"]


def kind_with(c: str, ignore_punctuation: bool = False) -> Kind:
    if c.isalnum() or c == "_":
        return "word"

    if c.isspace():
        return "whitespace"

    if ignore_punctuation:
        return "word"
    return "punctuation"


def kind2int(k: Kind) -> int:
    if k == "whitespace":
        return 0
    if k == "punctuation":
        return 1
    if k == "word":
        return 2
    raise ValueError(f"Unknown kind: {k}")


def int2kind(i: int) -> Kind:
    if i == 0:
        return "whitespace"
    if i == 1:
        return "punctuation"
    if i == 2:
        return "word"
    raise ValueError(f"Unknown int: {i}")
