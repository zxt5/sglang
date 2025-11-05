import time
import json
import requests


def run_once():
    response = requests.post(
        "http://localhost:36001/generate",
        json={
            "text": """
# Implementation and test cases of DFA in python

import
""".strip(),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 600,
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
            print(output[prev:], end="", flush=True)
            prev = len(output)


start_time = time.time()
for _ in range(1):
    run_once()
end_time = time.time()
print()
print(f"Total time: {end_time - start_time:.2f} seconds")
