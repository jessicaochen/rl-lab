# ruff: noqa: E501
"""Preprocess an R2E-Gym dataset into uni-agent's SWE task parquet format.

Rows carry the canonical Docker Hub image ref (`docker_image`, e.g.
`namanjain12/orange3_final:<sha>`); mapping to a registry cache is the sandbox
provider's job at run time (uni-agent `image_map`), never baked into the data.

Example::

    python -m rlbench_verl_tasks.r2e_gym.preprocess --local-save-dir ~/data/r2e --max-instances 64
"""

import argparse
import os

from datasets import load_dataset

DATA_SOURCE = "R2E-Gym/R2E-Gym-Subset"


def build_r2e_gym(dataset_name: str = DATA_SOURCE, split: str = "train", max_instances: int | None = None):
    def process(example):
        metadata = {
            "commit_hash": example.get("commit_hash", ""),
            "expected_output_json": example["expected_output_json"],
            "parsed_commit_content": example.get("parsed_commit_content", ""),
        }
        task_config = {
            "name": "r2e_gym",
            "sandbox": {"image": example["docker_image"]},
            "metadata": metadata,
        }
        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": example["problem_statement"]}],
            "extra_info": {
                "tools_kwargs": {"task": task_config},
            },
        }

    print(f"Loading {dataset_name} ({split}) from huggingface...", flush=True)
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} raw instances", flush=True)
    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)
    return dataset.map(process, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/r2e_gym")
    parser.add_argument("--dataset", default=DATA_SOURCE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    dataset = build_r2e_gym(args.dataset, args.split, args.max_instances)
    out_path = f"{save_dir}/r2e_gym.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
