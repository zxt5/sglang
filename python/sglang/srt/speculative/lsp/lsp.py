import os
import asyncio
from pathlib import Path
from typing import TypedDict

import lsp_types
from lsp_types.pyright.backend import PyrightBackend
from lsp_types.types import Position

from sglang.srt.speculative.lsp.buffer import Buffer
from sglang.srt.speculative.lsp.char_classifier import Kind


class CompletionCandidate(TypedDict):
    start: int
    end: int
    newtext: str


def filter_label(word: str, kind: Kind, label: str):
    if kind == "word":
        return label.startswith(word)
    return True


class LanguageClient:
    def __init__(self, initial_code: str):
        self.buffer = Buffer(initial_code)
        self.session = None

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

        completions = await self.session.get_completion(pos)
        completions = [
            item
            for item in completions["items"]
            if filter_label(word, wordkind, item["label"])
        ]
        completions.sort(key=lambda x: x["sortText"])

        res = []
        for item in completions:
            # to replace [start, end) with newtext
            entry = CompletionCandidate(
                start=start,
                end=end,
                newtext=item["label"],
            )
            res.append(entry)
        return res

    async def update_code(self, new_code: str):
        self.buffer = Buffer(new_code)
        if self.session is not None:
            await self.session.update_code(new_code)


if __name__ == "__main__":

    async def main():
        code = """
def add(x0, x1):
    return x
        """.strip()

        client = LanguageClient(initial_code="")
        await client.start(base_path=Path("."))
        await client.update_code(code)

        for pos in [Position(line=1, character=12), None]:
            res = await client.get_completion(pos)
            print([x["newtext"] for x in res])

    asyncio.run(main())
