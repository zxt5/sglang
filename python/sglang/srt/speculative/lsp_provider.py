from typing import List, Tuple, Protocol, Optional, Sequence, Dict
import numpy as np   

class LSPDraftProvider(Protocol):
    def batch_get(
        self,
        batch_context_tokens: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def reset(self) -> None:  # optional
        ...


class FakeLSPProvider:
    def __init__(self, pad_token_id: int = 0,
                 fixed_branches: Optional[List[List[int]]] = None,
                 encode: Optional[callable] = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.encode = encode
        self.fixed_branches = fixed_branches

    def reset(self) -> None:
        # stateless
        return

    def _build_rooted_bfs_order(
        self,
        branches: List[List[int]],
        capacity_without_root: int,
    ) -> List[Tuple[int, int]]:
        order: List[Tuple[int, int]] = []
        if capacity_without_root <= 0:
            return order
        max_depth = max((len(seq) for seq in branches), default=0)
        for d in range(max_depth):
            for b, seq in enumerate(branches):
                if d < len(seq):
                    order.append((b, d))
                if len(order) >= capacity_without_root:
                    return order
        return order

    def _pack_one_bfs(
        self,
        last_token: int,
        branches: List[List[int]],
        draft_token_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        last_token = int(last_token)
        capacity = max(draft_token_num - 1, 0) # slots after ROOT
        order = self._build_rooted_bfs_order(branches, capacity)

        # flat tokens with ROOT first
        flat_tokens: List[int] = [last_token]
        node_index: Dict[Tuple[int, int], int] = {}
        for k, (b, d) in enumerate(order, start=1): # start after ROOT
            tok = int(branches[b][d])
            flat_tokens.append(tok)
            node_index[(b, d)] = k
        if len(flat_tokens) < draft_token_num:
            flat_tokens.extend([self.pad_token_id] * (draft_token_num - len(flat_tokens)))
        tokens = np.asarray(flat_tokens[:draft_token_num], dtype=np.int64)

        # build (draft x draft) mask
        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.bool_)
        # ROOT sees itself
        if draft_token_num > 0:
            mask[0, 0] = True
        # Continuation nodes: see ROOT + same-branch ancestors
        for (b, d), i in node_index.items():
            # attend ROOT
            mask[i, 0] = True
            mask[i, i] = True
            # attend (b, dd) for dd < d if those nodes exist
            for dd in range(d):
                j = node_index.get((b, dd))
                if j is not None and j < i:
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

        for i, ctx in enumerate(batch_context_tokens):
            last_token = ctx[-1] if len(ctx) > 0 else int(self.pad_token_id)
            branches = self.get_drafts_for_context(ctx)
            toks, mflat = self._pack_one_bfs(last_token, branches, draft_token_num)
            out_tokens[i, :] = toks
            out_masks[i, :] = mflat

        return out_tokens.reshape(-1), out_masks.reshape(-1)

        

    def get_drafts_for_context(self, ctx: Sequence[int]) -> List[List[int]]:
        if self.encode is not None:
            def tok(s: str) -> List[int]:
                return self.encode(s.strip(), add_special_tokens=False)
            a = tok("1 2 3")
            b = tok("4 5")
            return [a, b]
        else:
            return [
                [1001, 1002, 1003],
                [2001, 2002]
            ]


if __name__ == "__main__":
    provider = FakeLSPProvider()
    ctx = [42, 43, 44] # last_token = 44 => root
    branches = [[1, 2, 3], [4, 5, 6]]
    toks, mflat = provider._pack_one_bfs(last_token=ctx[-1], branches=branches, draft_token_num=6)
    print("tokens:", toks)
    print("mask:\n", mflat.reshape(6, 6).astype(int))