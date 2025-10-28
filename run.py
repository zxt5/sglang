import time
import requests, json

RUN_SERVER = False

if RUN_SERVER:
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    enable_spec = True
    model = "Qwen/Qwen3-30B-A3B"

    if enable_spec:
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model-path {model} \
            --host 0.0.0.0 --port 36001 --log-level warning \
            --mem-fraction-static 0.9 \
            --speculative-algorithm NGRAM \
            --speculative-num-draft-tokens 16
        """
        )
    else:
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model-path {model} \
            --mem-fraction-static 0.9 \
            --host 0.0.0.0 --port 36001 --log-level warning
        """
        )

    wait_for_server(f"http://localhost:{port}")
    print("Server is up and running!")

def run_once():
    response = requests.post(
        "http://localhost:36001/generate",
        json={
            "text": """
# Implementation and test cases of DFA in Python

class
""".strip(),
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
            print(output[prev:], end="", flush=True)
            prev = len(output)

start_time = time.time()
for _ in range(1):
    run_once()
end_time = time.time()
print()
print(f"Total time: {end_time - start_time:.2f} seconds")

if RUN_SERVER:
    server_process.terminate()