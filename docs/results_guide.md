# Results Guide

The `results/` directory contains raw and aggregated outputs from the personalization-depth study.

The standard run folder naming pattern is:

```text
<dataset>_alpha<alpha_code>_seed<seed>/
```

Example:

```text
cifar10_alpha03_seed0/
```

Typical files include:

- `config.json`: experiment configuration
- `fedavg_metrics.json`: global FedAvg evaluation
- `oracle_results.json`: per-client personalization results
- `oracle_summary.json`: oracle routing summary
- `summary.json`: aggregated run summary

The script `scripts/aggregate_results.py` scans these outputs and builds a compact benchmark summary table.
