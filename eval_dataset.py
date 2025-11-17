import time
import json
import requests
from datasets import load_dataset
from pprint import pprint


def run_once(prompt: str):
    response = requests.post(
        "http://localhost:36001/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 300,
            },
            "stream": True,
        },
        stream=True,
        timeout=None,
    )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            output = data["text"]
            if data['extra']['accepted_draft_tokens'] > 0:
                print("\n[Accepted draft tokens: {}]\n".format(
                    data['extra']['accepted_draft_tokens']
                ), end="", flush=True)
            print(output[prev:], end="", flush=True)
            prev = len(output)


def main():
    ds = load_dataset("openai/openai_humaneval")
    for sample in ds["test"]:
        prompt = sample["prompt"]
        run_once(prompt)

main()
