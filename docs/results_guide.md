# Results Guide

The `results/` directory stores raw and aggregated outputs from the personalization-depth experiments.

Run folders follow the pattern:

```text
<dataset>_alpha<alpha_code>_seed<seed>/
```

Typical files include:

- `config.json`: run configuration
- `fedavg_metrics.json`: global model evaluation
- `oracle_results.json`: client-level policy results
- `oracle_summary.json`: oracle routing summary
- `summary.json`: aggregated run summary
