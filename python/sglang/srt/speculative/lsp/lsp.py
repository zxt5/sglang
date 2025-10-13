import os
import asyncio
from pathlib import Path
from typing import TypedDict

import lsp_types
from lsp_types import types
from lsp_types.pyright.backend import PyrightBackend
from lsp_types.types import Position

from sglang.srt.speculative.lsp.buffer import Buffer
from sglang.srt.speculative.lsp.char_classifier import Kind


class CompletionCandidate(TypedDict):
    start: int
    end: int
    newtext: str
    kind: int


def filter_label(word: str, kind: Kind, label: str):
    if kind == "word":
        return label.startswith(word)
    return True


def completion_item_kind2priority(kind: int) -> int:
    # from LSP spec
    # https://microsoft.github.io/language-server-protocol/specifications/specification-current/#textDocument_completion
    priority_map = {
        2: 1,  # Method
        3: 1,  # Function
        5: 1,  # Field
        6: 1,  # Variable
        10: 1,  # Property
        4: 2,  # Constructor
        7: 2,  # Class
        8: 2,  # Interface
        9: 3,  # Module
        11: 4,  # Unit
        12: 4,  # Value
        13: 4,  # Enum
        14: 4,  # Keyword
        20: 5,  # EnumMember
        21: 5,  # Constant
        22: 5,  # Struct
        23: 5,  # Event
        24: 5,  # Operator
        25: 5,  # TypeParameter
        16: 6,  # Color
        17: 6,  # File
        18: 6,  # Reference
        19: 6,  # Folder
        1: 30,  # Text
        15: 30,  # Snippet
    }
    return priority_map.get(kind, 100)  # default low priority


class LanguageClient:
    def __init__(self, initial_code: str, max_completions: int = 4):
        self.max_completions = max_completions
        self.buffer = Buffer(initial_code)
        self.session = None
        self.prev_word_start = None
        self.prev_completions = []

    async def start(self, base_path: Path | str):
        if isinstance(base_path, str):
            base_path = Path(base_path)

        self.session = await lsp_types.Session.create(
            PyrightBackend(),
            base_path=base_path,
            initial_code=self.buffer.text,
        )

    async def get_completion(
        self, pos: Position | None = None
    ) -> list[CompletionCandidate]:
        if self.session is None:
            raise RuntimeError("Session not started. Call start() first.")

        if pos is None:
            # use the ending
            offset = len(self.buffer.text)
            pos = self.buffer.offset2pos(offset)
        else:
            offset = self.buffer.pos2offset(pos)

        start, end, wordkind = self.buffer.surrounding_word(offset)
        word = self.buffer.text[start:end]

        if wordkind != "word" and word != '.':
            self.prev_word_start = None
            self.prev_completions = []
            return []

        if self.prev_word_start == start and self.prev_completions is not None:
            completions = [
                item
                for item in self.prev_completions
                if filter_label(word, wordkind, item["newtext"])
            ]
            return completions

        completions = await self.session.get_completion(pos)
        completions = [
            item
            for item in completions["items"]
            if filter_label(word, wordkind, item["label"])
        ]
        completions.sort(
            key=lambda x: (
                completion_item_kind2priority(x["kind"]),
                -len(x["label"]),
                x["sortText"],
            )
        )

        res = []
        for item in completions:
            # to replace [start, end) with newtext
            entry = CompletionCandidate(
                start=start, end=end, newtext=item["label"], kind=item["kind"]
            )
            res.append(entry)

        self.prev_word_start = start
        self.prev_completions = res

        return res[: self.max_completions]

    async def update_code(self, new_code: str):
        self.buffer = Buffer(new_code)
        if self.session is not None:
            await self.session.update_code(new_code)


if __name__ == "__main__":
    import time

    async def main():
        code1 = """
def add(x0, x1):
    return x
        """.strip()
        code2 = """
def a1_very_long_function_name():
    print("Hello, World!")

def another_function_that_calls_a_very_long_function_name():
    a
        """.strip()

        client = LanguageClient(initial_code="")
        await client.start(base_path=Path("."))

        for i, pos in enumerate(
            [
                Position(line=1, character=12),
                None,
            ]
            * 5
        ):
            start_time = time.time()
            await client.update_code(code1 if i % 2 == 0 else code2)
            res = await client.get_completion(pos)
            print([x["newtext"] for x in res])
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.3f} seconds")

            start_time = time.time()
            await client.update_code((code1 if i % 2 == 0 else code2) + "1")
            res = await client.get_completion(pos)
            print([x["newtext"] for x in res])
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.3f} seconds")

    asyncio.run(main())
