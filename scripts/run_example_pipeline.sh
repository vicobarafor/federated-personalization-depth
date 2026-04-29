#!/usr/bin/env bash
set -e

# Example high-level reproduction commands.
# Full sweeps can be expensive; these commands document the intended entry points.

python src/experiments/fedavg_train.py --dataset cifar10 --alpha 0.3 --seed 0
python src/experiments/personalization_oracle.py --dataset cifar10 --alpha 0.3 --seed 0
python src/experiments/selector_benchmark.py
