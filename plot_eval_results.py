import os
import json
import re
import subprocess
import sys
from collections import defaultdict

try:
    import scienceplots
except ImportError:
    print("scienceplots not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scienceplots"])

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename):
    match = re.search(r"eval-humaneval-(\d+)-([a-z-]+)-\db_final.json", filename)
    if match:
        return match.group(1), match.group(2)
    
    match = re.search(r"eval-humaneval-(\d+)-(.+?)-7b_final.json", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def plot_evaluation_results(results_dir="final_stats", output_dir="eval_results_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    grouped_data = defaultdict(lambda: defaultdict(dict))
    for f in files:
        with open(os.path.join(results_dir, f), "r") as fp:
            data = json.load(fp)
            token_size, method = parse_filename(f)
            if method == "lsp-ts":
                method = "lspec\n(lsp+ts)"
            if token_size and method:
                for metric, value in data.items():
                    if metric != "id":
                        grouped_data[token_size][metric][method] = value

    plt.style.use(['science', 'ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 20

    metrics = ["ttft", "tpot", "latency", "n_accepted_tokens", "n_generated_tokens"]

    for token_size, metric_data in grouped_data.items():
        for metric in metrics:
            if not metric_data[metric]:
                continue
            methods = list(metric_data[metric].keys())
            values = list(metric_data[metric].values())
            # sort methods and values based on methods
            methods, values = zip(*sorted(zip(methods, values)))

            plt.figure(figsize=(8, 6))
            bar_width = 0.5
            plt.bar(np.arange(len(methods)), values, width=bar_width)

            plt.ylabel(metric)
            plt.title(f"Comparison of {metric} (Max Tokens: {token_size})")
            plt.xticks(np.arange(len(methods)), methods, rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"{token_size}_{metric}_comparison.pdf"))
            plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plot_evaluation_results()
