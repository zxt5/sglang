import os
import glob
import numpy as np
from pydantic import BaseModel
from eval_dataset import MetricsList


class FinalMetrics(BaseModel):
    id: str
    ttft: float
    tpot: float
    latency: float
    n_accepted_tokens: float
    n_generated_tokens: float


def collect_stats(stats_file: str) -> MetricsList:
    with open(stats_file, "r") as f:
        metrics = MetricsList.model_validate_json(f.read())
    return metrics


def main():
    os.makedirs("final_stats", exist_ok=True)

    for path in glob.glob("eval_results/*.json"):
        metrics = collect_stats(path)

        res = FinalMetrics(
            id=path.split("/")[-1].replace(".json", ""),
            ttft=0,
            tpot=0,
            latency=0,
            n_accepted_tokens=0,
            n_generated_tokens=0,
        )

        for m in metrics.metrics:
            res.ttft += m.time_per_chunk[0]
            res.tpot += m.total_time / m.n_generated_tokens
            res.latency += m.total_time
            res.n_generated_tokens += m.n_generated_tokens
            res.n_accepted_tokens += np.sum(m.n_accepted_draft_tokens_per_chunk).item()

        res.ttft /= len(metrics.metrics)
        res.tpot /= len(metrics.metrics)
        res.latency /= len(metrics.metrics)
        res.n_accepted_tokens /= len(metrics.metrics)
        res.n_generated_tokens /= len(metrics.metrics)

        with open(f"final_stats/{res.id}_final.json", "w") as f:
            f.write(res.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
