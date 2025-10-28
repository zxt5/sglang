model=Qwen/Qwen3-30B-A3B
# model=qwen/qwen2.5-0.5b-instruct

python3 -m sglang.launch_server --model-path $model \
        --host 0.0.0.0 --port 36001 --log-level warning \
        --mem-fraction-static 0.9 \
        --speculative-algorithm NGRAM \
        --speculative-num-draft-tokens 16