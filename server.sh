#!/bin/bash

lang=python
# model=Qwen/Qwen3-30B-A3B
# model=Qwen/Qwen2.5-Coder-7B-Instruct
# model=qwen/qwen2.5-0.5b-instruct
# model=deepseek-ai/deepseek-coder-33b-instruct
model="codellama/CodeLlama-34b-hf"

python3 -m sglang.launch_server --model-path $model \
        --host 0.0.0.0 --port 36001 --log-level warning \
        --mem-fraction-static 0.9 \
        --enable-metrics \
        --enable-metrics-for-all-schedulers \
        --show-time-cost \
        --speculative-algorithm LSP \
        --speculative-num-draft-tokens 16 \
        --speculative-lsp-lang $lang \
        --speculative-lsp-only-lsp

# python3 -m sglang.launch_server --model-path $model \
#         --host 0.0.0.0 --port 36001 --log-level warning \
#         --mem-fraction-static 0.9 \
#         --enable-metrics \
#         --enable-metrics-for-all-schedulers \
#         --show-time-cost \
#         --speculative-algorithm NGRAM \
#         --speculative-num-draft-tokens 16


