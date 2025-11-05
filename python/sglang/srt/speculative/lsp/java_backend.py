from __future__ import annotations

import json
import typing as t
from pathlib import Path

import lsp_types
from lsp_types import types
from lsp_types.process import ProcessLaunchInfo
from lsp_types.session import LSPBackend


class JavaBackend(LSPBackend):
    def write_config(self, base_path: Path, options: t.Mapping) -> None:
        main_path = base_path / "src" / "Main.java"
        main_path.parent.mkdir(parents=True, exist_ok=True)
        if not main_path.exists():
            main_path.write_text("")

        config_path = base_path / "config"
        config_path.mkdir(parents=True, exist_ok=True)

        ws = base_path / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

    def create_process_launch_info(
        self, base_path: Path, options: t.Mapping
    ) -> ProcessLaunchInfo:
        return ProcessLaunchInfo(
            cmd=[
                "jdtls",
            ],
            cwd=base_path,
        )

    def get_lsp_capabilities(self) -> types.ClientCapabilities:
        return {
            "textDocument": {
                "completion": {
                    "completionItem": {
                        "snippetSupport": True,
                        "documentationFormat": [
                            lsp_types.MarkupKind.Markdown,
                            lsp_types.MarkupKind.PlainText,
                        ],
                        "commitCharactersSupport": True,
                        "labelDetailsSupport": True,
                    },
                    "insertTextMode": lsp_types.InsertTextMode.AsIs,
                    "completionItemKind": {
                        "valueSet": [
                            lsp_types.CompletionItemKind(x) for x in list(range(1, 26))
                        ]
                    },
                    "completionList": {
                        "itemDefaults": [
                            "commitCharacters",
                            "editRange",
                            "insertTextFormat",
                            "insertTextMode",
                        ]
                    },
                },
            }
        }

    def get_workspace_settings(
        self, options: t.Mapping
    ) -> types.DidChangeConfigurationParams:
        """Get workspace settings for didChangeConfiguration"""
        return {"settings": options}
