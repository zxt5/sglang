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
        self.encode = encode
        self.fixed_branches = fixed_branches

    def reset(self) -> None:
        # stateless
        return

    # def _pack_one(self, branches: List[List[int]], draft_token_num: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """Pack multiple branches into a flat token array and a (draft x draft) mask.

    #     We assign contiguous indices to each branch in order, truncate/pad to
    #     `draft_token_num`, and build a block-diagonal, strictly lower-triangular
    #     visibility mask per branch. Rows are queries, columns are keys.
    #     """
    #     # 1) flatten with truncation and remember branch-local indices
    #     flat_tokens: List[int] = []
    #     branch_index_ranges: List[Tuple[int, int]] = []  # [start, end) in flat space
    #     for seq in branches:
    #         if not seq:
    #             continue
    #         start = len(flat_tokens)
    #         room = draft_token_num - start
    #         if room <= 0:
    #             break
    #         take = seq[:room]
    #         flat_tokens.extend(int(x) for x in take)
    #         end = start + len(take)
    #         branch_index_ranges.append((start, end))
    #     # pad to fixed length
    #     if len(flat_tokens) < draft_token_num:
    #         flat_tokens.extend([self.pad_token_id] * (draft_token_num - len(flat_tokens)))

    #     tokens = np.asarray(flat_tokens[:draft_token_num], dtype=np.int64)

    #     # 2) build (draft x draft) visibility mask
    #     mask = np.zeros((draft_token_num, draft_token_num), dtype=np.bool_)
    #     for (s, e) in branch_index_ranges:
    #         # strictly lower-triangular within [s, e)
    #         # i can attend j iff s <= j < i < e
    #         for i in range(s, e):
    #             for j in range(s, i):
    #                 mask[i, j] = True
    #     # PAD rows remain all False
    #     return tokens, mask.reshape(-1)

    def _pack_one_bfs(
        self,
        branches: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        BFS 打包：按层展开草案。
        层0: 每个分支的第一个 token（若有）
        层1: 每个分支的第二个 token（若有）
        ...
        这样重建出来的 next_sibling/next_token 链接就和 NGRAM 的 BFS 语义一致。
        对于可见性：允许每个结点看到“本分支上更早层”的所有祖先（祖先闭包）。
        """
        # 1) 生成 BFS 顺序的节点列表（记录 (branch_id, depth) -> flat_idx）
        order: List[Tuple[int, int]] = []
        max_depth = max((len(seq) for seq in branches), default=0)
        for d in range(max_depth):
            for b, seq in enumerate(branches):
                if d < len(seq):
                    order.append((b, d))
                    if len(order) >= draft_token_num:
                        break
            if len(order) >= draft_token_num:
                break

        # 2) 按 order 取 token，pad 到固定长度
        flat_tokens: List[int] = []
        node_index: dict[Tuple[int, int], int] = {}
        for idx, (b, d) in enumerate(order):
            tok = int(branches[b][d])
            flat_tokens.append(tok)
            node_index[(b, d)] = idx
        if len(flat_tokens) < draft_token_num:
            flat_tokens.extend([self.pad_token_id] * (draft_token_num - len(flat_tokens)))

        tokens = np.asarray(flat_tokens[:draft_token_num], dtype=np.int64)

        # 3) 构建 (draft x draft) mask：祖先闭包（仅限本分支）
        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.bool_)
        for (b, d), i in node_index.items():
            # 允许看同一分支上更早深度的所有祖先
            for dd in range(d):
                j = node_index.get((b, dd), None)
                if j is not None and j < i:
                    mask[i, j] = True
        # 注意：PAD 行列保持全 False

        print("pack_one_bfs", branches, draft_token_num)
        print("tokens", tokens)
        print("mask", mask)

        return tokens, mask.reshape(-1)


    def batch_get(
        self,
        batch_tokens: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:


        print("No matter what speculative algorithm, the batch_tokens should be the same")
        print("batch_tokens: ", batch_tokens)


        bs = len(batch_tokens)
        out_tokens = np.empty((bs, draft_token_num), dtype=np.int64)
        out_masks = np.empty((bs, draft_token_num * draft_token_num), dtype=np.bool_)

        # Note: we ignore context in this fake provider, but keep the signature.
        for i in range(bs):
            branches = self.get_drafts_for_context(batch_tokens[i])
            toks, mflat = self._pack_one_bfs(branches, draft_token_num)
            out_tokens[i, :] = toks
            out_masks[i, :] = mflat

        return out_tokens.reshape(-1), out_masks.reshape(-1)

    def get_drafts_for_context(self, ctx: Sequence[int]) -> List[List[int]]:


        def tok(s: str) -> List[int]:
            return self.encode(s, add_special_tokens=False)

        # 分支A：", 1, 2, 3"
        a = tok(", 1") + tok(", 2") + tok(", 3")
        # 分支B：", 4, 5"
        b = tok(", 4") + tok(", 5")

        # 如果太长，会在 _pack_one 里按 draft_token_num 截断

        print("get_drafts_for_context", [a, b])
        
        return [a, b]