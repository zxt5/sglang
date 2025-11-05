# sglang/srt/speculative/lsp_worker.py
#
# A drop-in speculative worker that reuses the NGRAM verification path
# but sources drafts from a pluggable LSP provider. For bootstrapping,
# we include a FakeLSPProvider that returns fixed drafts so you can
# validate the end-to-end flow without wiring any external service.

from __future__ import annotations

import time
import logging
from typing import List, Optional, Tuple, Protocol
from transformers import AutoTokenizer

import numpy as np
import torch

from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.lsp_info import LspVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


from sglang.srt.speculative.lsp_provider import LSPProvider

logger = logging.getLogger(__name__)

USE_FULL_MASK = True  # keep parity with ngram_worker default


class LSPWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        *,
        pad_token_id: Optional[int] = 0,
        max_match_window_size: Optional[int] = None,
    ) -> None:
        self.lang = server_args.speculative_lsp_lang
        self.target_model_id = server_args.model_path
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_id)

        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size

        # keep the same knobs as ngram
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_match_window_size: int = (
            int(max_match_window_size)
            if max_match_window_size is not None
            else int(server_args.speculative_ngram_max_match_window_size)
        )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.pad_token_id = int(pad_token_id or 0)
        self.lsp_provider = LSPProvider(
            lang=self.lang,
            tokenizer=self.target_tokenizer,
            pad_token_id=self.pad_token_id,
        )

        self._init_preallocated_tensors()

    # ---------------- infra ----------------
    def clear_cache_pool(self) -> None:
        if hasattr(self.lsp_provider, "reset"):
            try:
                self.lsp_provider.reset()
            except Exception as e:
                logger.warning(f"LSP provider reset() raised: {e}")

    @staticmethod
    def get_context_tokens(
        seq1: List[int], seq2: List[int], max_match_window_size: int
    ) -> List[int]:
        # zxt: feel free to change this function for LSP spec
        return seq1 + seq2

        # seq2_len = len(seq2)
        # if seq2_len >= n:
        #     return seq2[-n:]

        # need_from_seq1 = n - seq2_len
        # return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self) -> None:
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch: List[torch.Tensor] = []
        self.tree_mask_batch: List[torch.Tensor] = []
        self.retrieve_indexes_batch: List[torch.Tensor] = []
        self.retrive_next_token_batch: List[torch.Tensor] = []
        self.retrive_next_sibling_batch: List[torch.Tensor] = []
        self.positions_batch: List[torch.Tensor] = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    # --------------- core: obtain drafts from provider ---------------
    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> Tuple[np.ndarray, np.ndarray]:
        bs = batch.batch_size()

        batch_tokens = []
        for req in batch.reqs:
            check_token = self.get_context_tokens(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(check_token)

        req_drafts, mask = self.lsp_provider.batch_get(
            batch_tokens, self.draft_token_num
        )

        expected_tokens = bs * self.draft_token_num
        expected_mask = bs * self.draft_token_num * self.draft_token_num
        assert isinstance(req_drafts, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert req_drafts.size == expected_tokens, (
            f"LSP returned {req_drafts.size=}, expected {expected_tokens=}"
        )
        assert mask.size == expected_mask, (
            f"LSP returned {mask.size=}, expected {expected_mask=}"
        )
        if req_drafts.dtype != np.int64:
            req_drafts = req_drafts.astype(np.int64, copy=False)
        if mask.dtype != np.bool_:
            mask = mask.astype(np.bool_, copy=False)

        return req_drafts, mask

    # --------------- same prep path as ngram ---------------
    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # copied from ngram_worker
        if USE_FULL_MASK:
            tree_mask: List[torch.Tensor] = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix = torch.ones(
                    (self.draft_token_num, seq_len - 1), device=self.device
                )
                intra = torch.from_numpy(mask[i]).to(self.device)
                req_mask = torch.cat((prefix, intra), dim=1).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.LSP
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = LspVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )

        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _feedback_to_lsp(self, batch: ScheduleBatch) -> None:
        # no-op for fake provider; hook for real provider learning/signals
        return

    # --------------- main entry ---------------
    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            verify_input = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size
            )

            if num_accepted_tokens > 0:
                res_tokens = self.target_tokenizer.decode(next_token_ids[:num_accepted_tokens].tolist())
                print(f"[LSPWorker] next {num_accepted_tokens} tokens: {res_tokens}")

            # print(f"[LSPWorker] accepted {num_accepted_tokens}/{self.draft_token_num} drafts")
            # res_tokens = self.target_tokenizer.decode(next_token_ids[0].tolist())
            # print(f"[LSPWorker] next tokens: {res_tokens}")

            self._feedback_to_lsp(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            # fallback: plain decode
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
        )
