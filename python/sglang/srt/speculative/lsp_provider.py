# sglang/srt/speculative/lsp_provider.py
from typing import List, Tuple, Protocol, Optional, Sequence
import numpy as np   

class LSPDraftProvider(Protocol):
    """Interface for providing speculative drafts.

    Returns flattened tokens and a per-request (draft x draft) boolean mask
    describing intra-draft visibility (which draft positions can attend which
    previous draft positions). The mask must be flattened row-major when returned.
    """

    def batch_get(
        self,
        batch_context_tokens: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def reset(self) -> None:  # optional
        ...


class FakeLSPProvider:
    """A minimal provider that returns fixed drafts per request.

    By default, it produces two linear branches:
      branch A: [1001, 1002, 1003]
      branch B: [2001, 2002]
    These are packed into a length-`draft_token_num` flat list per request.

    The intra-draft mask is block-diagonal lower-triangular so each token
    only attends earlier tokens within its own branch. PAD slots are fully
    masked (no incoming visibility).
    """

    def __init__(self, pad_token_id: int = 0,
                 fixed_branches: Optional[List[List[int]]] = None,
                 encode: Optional[callable] = None) -> None:
        self.pad_token_id = int(pad_token_id)
        # default fixed drafts if none provided
        # self.fixed_branches: List[List[int]] = fixed_branches or [
        #     [1001, 1002, 1003],  # branch A
        #     [2001, 2002],        # branch B
        # ]
        self.encode = encode  # 新增
        self.fixed_branches = fixed_branches

    def reset(self) -> None:
        # stateless
        return

    def _pack_one(self, branches: List[List[int]], draft_token_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pack multiple branches into a flat token array and a (draft x draft) mask.

        We assign contiguous indices to each branch in order, truncate/pad to
        `draft_token_num`, and build a block-diagonal, strictly lower-triangular
        visibility mask per branch. Rows are queries, columns are keys.
        """
        # 1) flatten with truncation and remember branch-local indices
        flat_tokens: List[int] = []
        branch_index_ranges: List[Tuple[int, int]] = []  # [start, end) in flat space
        for seq in branches:
            if not seq:
                continue
            start = len(flat_tokens)
            room = draft_token_num - start
            if room <= 0:
                break
            take = seq[:room]
            flat_tokens.extend(int(x) for x in take)
            end = start + len(take)
            branch_index_ranges.append((start, end))
        # pad to fixed length
        if len(flat_tokens) < draft_token_num:
            flat_tokens.extend([self.pad_token_id] * (draft_token_num - len(flat_tokens)))

        tokens = np.asarray(flat_tokens[:draft_token_num], dtype=np.int64)

        # 2) build (draft x draft) visibility mask
        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.bool_)
        for (s, e) in branch_index_ranges:
            # strictly lower-triangular within [s, e)
            # i can attend j iff s <= j < i < e
            for i in range(s, e):
                for j in range(s, i):
                    mask[i, j] = True
        # PAD rows remain all False
        return tokens, mask.reshape(-1)

    def batch_get(
        self,
        batch_context_tokens: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bs = len(batch_context_tokens)
        out_tokens = np.empty((bs, draft_token_num), dtype=np.int64)
        out_masks = np.empty((bs, draft_token_num * draft_token_num), dtype=np.bool_)

        # Note: we ignore context in this fake provider, but keep the signature.
        for i in range(bs):
            branches = self.get_drafts_for_context(batch_context_tokens[i])
            toks, mflat = self._pack_one(branches, draft_token_num)
            out_tokens[i, :] = toks
            out_masks[i, :] = mflat

        return out_tokens.reshape(-1), out_masks.reshape(-1)

    def get_drafts_for_context(self, ctx: Sequence[int]) -> List[List[int]]:
        if self.encode is None:
            # 兜底（不推荐）：老的硬编码分支
            return [[1001, 1002, 1003], [2001, 2002]]

        # 用 tokenizer.encode 生成更合理的草案（示例：数字+逗号）
        # LLaMA 系列通常需要前置空格才能取到“独立数字”token
        def tok(s: str) -> List[int]:
            return self.encode(s, add_special_tokens=False)

        # 分支A：", 1, 2, 3"
        a = tok(", 1") + tok(", 2") + tok(", 3")
        # 分支B：", 4, 5"
        b = tok(", 4") + tok(", 5")

        # 如果太长，会在 _pack_one 里按 draft_token_num 截断
        return [a, b]