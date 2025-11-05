lang=python
# model=Qwen/Qwen3-30B-A3B
model=Qwen/Qwen2.5-Coder-7B-Instruct
# model=qwen/qwen2.5-0.5b-instruct

python3 -m sglang.launch_server --model-path $model \
        --host 0.0.0.0 --port 36001 --log-level warning \
        --mem-fraction-static 0.9 \
        --speculative-algorithm LSP \
        --speculative-num-draft-tokens 16 \
        --speculative-lsp-lang $lang