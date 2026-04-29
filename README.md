# How Much Should Each Client Personalize?

## Adaptation Depth in Federated Learning

This repository contains the implementation, experimental artifacts, figures, and manuscript for a research study on **client-specific personalization depth in federated learning**.

The central research question is:

> How much of a shared model should each client personalize after federated training?

## Key Insight

Personalization depth is not globally optimal. Different clients can prefer different adaptation depths within the same federated run, and forcing all clients to use a single policy leaves measurable performance unrealized.

## Overview

Personalized federated learning is commonly motivated by client heterogeneity, but many systems still apply the same adaptation strategy to every client. This project studies **personalization depth** as a client-dependent decision.

We compare three post-hoc personalization policies:

- **Head-only fine-tuning**
- **Partial fine-tuning**
- **Full fine-tuning**

Experiments are conducted across five benchmark datasets and multiple Dirichlet non-IID regimes.

## Main Contributions

- Formulates client-specific personalization depth as a distinct federated learning decision problem.
- Evaluates head-only, partial, and full fine-tuning across multiple datasets and heterogeneity regimes.
- Quantifies oracle client-wise routing headroom beyond the strongest fixed policy.
- Analyzes client-level variation in preferred adaptation depth.
- Evaluates lightweight automatic selectors and shows that recovering oracle gains is nontrivial.

## Datasets

- CIFAR-10
- CIFAR-100
- SVHN
- FashionMNIST
- EMNIST-Balanced

Each dataset is evaluated under Dirichlet non-IID client partitions with multiple heterogeneity settings and random seeds.

## Repository Structure

```text
federated-personalization-depth/
├── src/
│   ├── data/                  # Dataset loading and client partitioning
│   ├── experiments/           # Experiment drivers and oracle/selector studies
│   ├── fl/                    # Federated learning and personalization modules
│   ├── models/                # Model definitions
│   └── utils/                 # Evaluation and helper utilities
├── configs/                   # Experiment configuration files
├── scripts/                   # Reproduction and execution helpers
├── results/                   # Raw and aggregated experiment outputs
├── results_FINAL_CAMERA_READY/ # Final paper-ready result package
├── figures/                   # Consolidated paper and appendix figures
├── analysis/                  # Analysis notes and diagnostics
├── docs/                      # Methodology and reproducibility documentation
├── notebooks/                 # Exploratory notebooks
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Key Results and Figures

### Winning Fixed Personalization Depth

![Winning fixed personalization depth](figures/winner_heatmap.png)

### Oracle Routing Headroom

![Oracle headroom](figures/oracle_headroom.png)

### Client-Level Depth Preferences

![Client depth distribution](figures/client_depth_distribution.png)

### Per-Client Oracle Gains

![Client oracle gain histogram](figures/client_oracle_gain_histogram.png)

### Oracle vs Fixed Policy Disagreement

![Oracle fixed disagreement](figures/oracle_fixed_disagreement_matrix.png)

### Oracle Win Margins

![Oracle win margin histogram](figures/oracle_win_margin_histogram.png)

Additional final figures are available in `figures/` and `results_FINAL_CAMERA_READY/`.

## Code Organization

### `src/data/`

Dataset loading and client partitioning utilities for the benchmark datasets used in the study.

### `src/fl/`

Federated learning and personalization components, including FedAvg, client-side adaptation, oracle-study utilities, personalization probes, and FSAL-style client variants.

### `src/experiments/`

Experiment entry points for FedAvg training, personalization-oracle evaluation, oracle regret analysis, selector benchmarking, learned selector experiments, and dynamic selector studies.

### `results/`

Raw and aggregated outputs from multi-dataset, multi-alpha, and multi-seed experiments.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FedAvg pretraining:

```bash
python src/experiments/fedavg_train.py --dataset cifar10 --alpha 0.3 --seed 0
```

Run personalization-oracle evaluation:

```bash
python src/experiments/personalization_oracle.py --dataset cifar10 --alpha 0.3 --seed 0
```

Run selector benchmarking:

```bash
python src/experiments/selector_benchmark.py
```

The repository includes stored results and paper-ready figures so that the main empirical claims can be inspected without rerunning the full experimental sweep.

## Limitations

- Oracle routing is an upper bound, not a deployable method.
- Lightweight selectors do not consistently recover oracle-level gains.
- Experiments focus on controlled benchmark datasets with synthetic Dirichlet partitions.
- Additional validation on real-world federated datasets would strengthen external validity.

## Paper

A paper manuscript is included in the project artifacts. This work is being prepared for external academic review and submission.

If `paper/main.pdf` is present, the manuscript can be opened directly from:

[paper/main.pdf](paper/main.pdf)

## Research Contribution

This project identifies **personalization depth** as a distinct axis of heterogeneity in federated learning. It shows that fixed personalization policies leave measurable performance unrealized and that adaptive depth selection is a nontrivial client-level decision problem.

## Author

Victor Obarafor

## License

MIT License
