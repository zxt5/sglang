import requests, json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 --port 36001 --log-level warning \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16
"""
)

wait_for_server(f"http://localhost:{port}")

print("Server is up and running!")

print()

while True:
    response = requests.post(
        f"http://localhost:{port}/generate",
        json={
            "text": """
def a_very_long_function_name():
    print("Hello, World!")

def another_function_that_calls_a_very_long_function_name():
    a
""".strip(),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 20,
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
