import os
import time
import asyncio
from pathlib import Path
from typing import List, TypedDict

import lsp_types
from lsp_types import types
from lsp_types.pyright.backend import PyrightBackend
from lsp_types.types import Position

from sglang.srt.speculative.lsp.buffer import Buffer
from sglang.srt.speculative.lsp.char_classifier import Kind
from sglang.srt.speculative.lsp.rust_backend import RustAnalayzerBackend
from sglang.srt.speculative.lsp.session import Session


class CompletionCandidate(TypedDict):
    start: int
    end: int
    newtext: str
    kind: int


def filter_label(word: str, kind: Kind, label: str):
    if kind == "word":
        return label.startswith(word)
    return True


def process_label(item: lsp_types.CompletionItem) -> str:
    label = item["label"]
    if "textEdit" in item:
        text_edit = item["textEdit"]
        if "newText" in text_edit:
            # extract common prefix of label and newText
            new_text = text_edit["newText"]
            i = 0
            m = min(len(label), len(new_text))
            while i < m and label[i] == new_text[i]:
                i += 1
            label = label[:i]

    return label


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
    def __init__(
        self, initial_code: str, max_completions: int = 8, lang: str = "python"
    ):
        self.max_completions = max_completions
        self.buffer = Buffer(initial_code)
        self.session = None
        self.prev_word_start = None
        self.prev_completions: List[CompletionCandidate] = []
        self.dirty = False
        self.increased_code = None

        self.lang = lang

    async def start(self, base_path: Path | str):
        if isinstance(base_path, str):
            base_path = Path(base_path)

        if self.lang in ["rust"]:
            self.session = await Session.create(
                RustAnalayzerBackend(),
                base_path=base_path,
                initial_code=self.buffer.text,
                document_uri=f"file://{base_path / 'src' / 'main.rs'}",
            )
            time.sleep(5)  # wait for rust-analyzer to be ready (indexing)
        elif self.lang in ["python"]:
            self.session = await Session.create(
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

        if wordkind != "word" and word not in [":", ".", ">"]:
            self.prev_word_start = None
            self.prev_completions = []
            return []

        if self.prev_word_start == start and self.prev_completions is not None:
            completions = [
                item
                for item in self.prev_completions
                if filter_label(word, wordkind, item["newtext"])
            ]
            for c in completions:
                c["end"] = end
            self.prev_completions = completions
            return completions[: self.max_completions]

        if self.dirty:
            if self.increased_code is not None:
                await self.session.update_code(
                    self.increased_code,
                    incremental_pos=self.buffer.offset2pos(
                        len(self.buffer.text) - len(self.increased_code)
                    ),
                )
                self.increased_code = None
            else:
                if self.lang in ["rust"]:
                    await self.session.close_document()
                    await self.session.open_document(self.buffer.text)
                else:
                    await self.session.update_code(self.buffer.text)
            self.dirty = False

        completions = await self.session.get_completion(pos)
        if completions is None:
            return []
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
                start=start, end=end, newtext=process_label(item), kind=item["kind"]
            )
            if entry["newtext"]:
                res.append(entry)

        self.prev_word_start = start
        self.prev_completions = res

        return res[: self.max_completions]

    async def update_code(self, new_code: str, incremental: bool = False):
        if incremental:
            if self.lang in ["rust"]:
                if self.increased_code is None:
                    self.increased_code = new_code
                else:
                    self.increased_code += new_code

            self.buffer.text += new_code
        else:
            self.increased_code = None
            self.buffer.text = new_code

        self.dirty = True


if __name__ == "__main__":
    import time

    async def test_pyright():
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
        code3 = """
class DFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

DFA(
    tra
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

        await client.update_code(code3)
        res = await client.get_completion()
        print([x["newtext"] for x in res])

    async def test_rust():
        code1 = """fn add(x0: i32, x1: i32) -> i32 {
    x
""".strip()
        code2 = """fn a1_very_long_function_name() {
    println!("Hello, World!");
}
fn another_function_that_calls_a_very_long_function_name() {
    a
""".strip()

        client = LanguageClient(initial_code="", lang="rust")
        await client.start(os.path.realpath("/data/h445xu/repo/temp"))
        time.sleep(5)
        await client.update_code(code1, True)
        res = await client.get_completion()
        print([x["newtext"] for x in res])

    asyncio.run(test_pyright())
    asyncio.run(test_rust())
