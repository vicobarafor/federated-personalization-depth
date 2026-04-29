
import os
import json
import csv
from collections import defaultdict
import statistics


def safe_mean(values):
    return statistics.mean(values) if len(values) > 0 else None


def safe_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_experiment_name(name):
    """
    Expected naming style:
      cifar10_alpha03_seed0
      cifar100_alpha03_seed2
    """
    parts = name.split("_")
    parsed = {
        "dataset": parts[0] if len(parts) > 0 else None,
        "alpha_tag": parts[1] if len(parts) > 1 else None,
        "seed_tag": parts[2] if len(parts) > 2 else None,
    }
    return parsed


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def aggregate_results(results_root):
    exp_dirs = []

    for item in os.listdir(results_root):
        full_path = os.path.join(results_root, item)
        if os.path.isdir(full_path):
            summary_path = os.path.join(full_path, "summary.json")
            oracle_summary_path = os.path.join(full_path, "oracle_summary.json")
            fedavg_metrics_path = os.path.join(full_path, "fedavg_metrics.json")
            if os.path.exists(summary_path) and os.path.exists(oracle_summary_path) and os.path.exists(fedavg_metrics_path):
                exp_dirs.append((item, full_path))

    raw_rows = []
    grouped = defaultdict(list)
    grouped_wins = defaultdict(lambda: defaultdict(int))

    for exp_name, exp_dir in sorted(exp_dirs):
        parsed = parse_experiment_name(exp_name)

        fedavg_metrics = load_json(os.path.join(exp_dir, "fedavg_metrics.json"))
        oracle_summary = load_json(os.path.join(exp_dir, "oracle_summary.json"))

        row = {
            "experiment_name": exp_name,
            "dataset": parsed["dataset"],
            "alpha_tag": parsed["alpha_tag"],
            "seed_tag": parsed["seed_tag"],
            "fedavg_acc": fedavg_metrics["accuracy"],
            "head_only_acc": oracle_summary["head_only_mean_acc"],
            "conv2_head_acc": oracle_summary["conv2_head_mean_acc"],
            "full_ft_acc": oracle_summary["full_ft_mean_acc"],
            "wins_head_only": oracle_summary["wins"]["head_only"],
            "wins_conv2_head": oracle_summary["wins"]["conv2_head"],
            "wins_full_ft": oracle_summary["wins"]["full_ft"],
        }
        raw_rows.append(row)

        group_key = (row["dataset"], row["alpha_tag"])
        grouped[group_key].append(row)

        grouped_wins[group_key]["head_only"] += row["wins_head_only"]
        grouped_wins[group_key]["conv2_head"] += row["wins_conv2_head"]
        grouped_wins[group_key]["full_ft"] += row["wins_full_ft"]

    summary_rows = []

    for group_key, rows in grouped.items():
        dataset, alpha_tag = group_key

        fedavg_vals = [r["fedavg_acc"] for r in rows]
        head_vals = [r["head_only_acc"] for r in rows]
        conv2_vals = [r["conv2_head_acc"] for r in rows]
        full_vals = [r["full_ft_acc"] for r in rows]

        summary_rows.append({
            "dataset": dataset,
            "alpha_tag": alpha_tag,
            "num_runs": len(rows),

            "fedavg_mean": safe_mean(fedavg_vals),
            "fedavg_std": safe_std(fedavg_vals),

            "head_only_mean": safe_mean(head_vals),
            "head_only_std": safe_std(head_vals),

            "conv2_head_mean": safe_mean(conv2_vals),
            "conv2_head_std": safe_std(conv2_vals),

            "full_ft_mean": safe_mean(full_vals),
            "full_ft_std": safe_std(full_vals),

            "wins_head_only_total": grouped_wins[group_key]["head_only"],
            "wins_conv2_head_total": grouped_wins[group_key]["conv2_head"],
            "wins_full_ft_total": grouped_wins[group_key]["full_ft"],
        })

    return raw_rows, summary_rows


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_aggregation(results_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    raw_rows, summary_rows = aggregate_results(results_root)

    raw_csv_path = os.path.join(output_dir, "raw_experiment_results.csv")
    summary_csv_path = os.path.join(output_dir, "aggregated_summary.csv")
    raw_json_path = os.path.join(output_dir, "raw_experiment_results.json")
    summary_json_path = os.path.join(output_dir, "aggregated_summary.json")

    if raw_rows:
        save_csv(
            raw_rows,
            raw_csv_path,
            fieldnames=[
                "experiment_name",
                "dataset",
                "alpha_tag",
                "seed_tag",
                "fedavg_acc",
                "head_only_acc",
                "conv2_head_acc",
                "full_ft_acc",
                "wins_head_only",
                "wins_conv2_head",
                "wins_full_ft",
            ],
        )
        save_json(raw_rows, raw_json_path)

    if summary_rows:
        save_csv(
            summary_rows,
            summary_csv_path,
            fieldnames=[
                "dataset",
                "alpha_tag",
                "num_runs",
                "fedavg_mean",
                "fedavg_std",
                "head_only_mean",
                "head_only_std",
                "conv2_head_mean",
                "conv2_head_std",
                "full_ft_mean",
                "full_ft_std",
                "wins_head_only_total",
                "wins_conv2_head_total",
                "wins_full_ft_total",
            ],
        )
        save_json(summary_rows, summary_json_path)

    return {
        "num_raw_rows": len(raw_rows),
        "num_summary_rows": len(summary_rows),
        "raw_csv_path": raw_csv_path,
        "summary_csv_path": summary_csv_path,
        "raw_json_path": raw_json_path,
        "summary_json_path": summary_json_path,
        "summary_rows": summary_rows,
    }
