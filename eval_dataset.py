import os
import time
import json
import requests
from datasets import load_dataset
from pprint import pprint
from pydantic import BaseModel


class Metrics(BaseModel):
    id: str
    chunks: list[str]
    time_per_chunk: list[float]
    total_time: float
    n_prompt_tokens: int
    n_generated_tokens: int
    n_accepted_draft_tokens_per_chunk: list[int]


class MetricsList(BaseModel):
    metrics: list[Metrics]


def run_once(id: str, prompt: str, max_new_tokens: int) -> Metrics:
    response = requests.post(
        "http://localhost:36001/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
            "stream": True,
        },
        stream=True,
        timeout=None,
    )

    metrics = Metrics(
        id=id,
        chunks=[],
        time_per_chunk=[],
        total_time=0.0,
        n_prompt_tokens=0,
        n_generated_tokens=0,
        n_accepted_draft_tokens_per_chunk=[],
    )

    start_time = init_time = time.time()
    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            metrics.time_per_chunk.append(time.time() - start_time)
            start_time = time.time()

            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))

            metrics.n_prompt_tokens = data["meta_info"]["prompt_tokens"]
            metrics.n_generated_tokens = data["meta_info"]["completion_tokens"]
            n = data["extra"]["accepted_draft_tokens"]
            metrics.n_accepted_draft_tokens_per_chunk.append(n if n is not None else 0)
            output = data["text"]
            metrics.chunks.append(output[prev:])
            prev = len(output)

    metrics.total_time = time.time() - init_time
    return metrics


def main():

    alg = "nospec" # "lsp" or "lsp-ts" or "ngram" or "nospec" or "ts"
    model = "32b"

    metrics_list = MetricsList(metrics=[])
    ds = load_dataset("openai/openai_humaneval")
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs("stack_results", exist_ok=True)


    def eval_stack(max_new_tokens: int):
        prompts_jsonl = "/home/x27zhou/lspec/prompts_python.jsonl"
        prompts = []
        with open(prompts_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item["prompt"])
        for i, prompt in enumerate(prompts):
            metrics = run_once(
                f"stack-{i}",
                prompt,
                max_new_tokens=max_new_tokens,
            )
            print(
                f"Completed sample {i + 1}, {metrics.total_time=:.2f}, {metrics.n_generated_tokens=}, {sum(metrics.n_accepted_draft_tokens_per_chunk)=}"
            )
            metrics_list.metrics.append(metrics)

        with open(f"stack_results/eval-stack-{max_new_tokens}-{alg}-{model}.json", "w") as f:
            f.write(metrics_list.model_dump_json(indent=2))

    # eval_stack(100)
    # eval_stack(300)


    def eval(max_new_tokens: int):
        for i, sample in enumerate(ds["test"]):
            prompt = sample["prompt"]
            metrics = run_once(f"humaneval-{i}", prompt, max_new_tokens=max_new_tokens)
            print(
                f"Completed sample {i + 1}, {metrics.total_time=:.2f}, {metrics.n_generated_tokens=}, {sum(metrics.n_accepted_draft_tokens_per_chunk)=}"
            )
            metrics_list.metrics.append(metrics)

        # save metrics list
        with open(f"eval_results/eval-humaneval-{max_new_tokens}-{alg}-{model}.json", "w") as f:
            f.write(metrics_list.model_dump_json(indent=2))

    eval(100)
    eval(300)


if __name__ == "__main__":
    main()
