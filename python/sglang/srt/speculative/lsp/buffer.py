from lsp_types.types import Position

from sglang.srt.speculative.lsp.char_classifier import kind_with, kind2int, int2kind


class Buffer:
    def __init__(self, text: str):
        self.text = text

    def surrounding_word(self, start: int):
        end = start

        next_chars = self.text[start : start + 128]
        prev_chars = list(reversed(self.text[max(0, start - 128) : start]))

        pkind = kind_with(prev_chars[0]) if len(prev_chars) > 0 else "whitespace"
        nkind = kind_with(next_chars[0]) if len(next_chars) > 0 else "whitespace"
        word_kind = int2kind(max(kind2int(pkind), kind2int(nkind)))

        for ch in prev_chars:
            if kind_with(ch) == word_kind and ch != "\n":
                start -= 1
            else:
                break

        for ch in next_chars:
            if kind_with(ch) == word_kind and ch != "\n":
                end += 1
            else:
                break

        return start, end, word_kind

    def offset2pos(self, offset: int):
        lines = self.text.split("\n")
        current_offset = 0
        for i, line in enumerate(lines):
            if current_offset + len(line) >= offset:
                return Position(line=i, character=offset - current_offset)
            current_offset += len(line) + 1  # +1 for the newline
        raise ValueError(f"Offset {offset} out of range")

    def pos2offset(self, pos: Position):
        lines = self.text.split("\n")
        if pos["line"] >= len(lines):
            raise ValueError(f"Line {pos['line']} out of range")
        line = lines[pos["line"]]
        if pos["character"] > len(line):
            raise ValueError(
                f"Character {pos['character']} out of range in line {pos['line']}"
            )
        return sum(len(lines[i]) + 1 for i in range(pos["line"])) + pos["character"]


if __name__ == "__main__":
    import os

    DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f"{DIR}/example/pyproject/main.py") as f:
        buffer = Buffer(f.read())

    pos = Position(line=24, character=10)
    offset = buffer.pos2offset(pos)
    assert buffer.text[offset - 2 : offset] == "se"
    assert buffer.offset2pos(offset) == pos
    start, end, wordkind = buffer.surrounding_word(offset)
    print(f"{wordkind=}")
    print(buffer.text[start:end])
    assert wordkind == "word"
    assert buffer.text[start:end] == "se"
