from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = ROOT / 'results' / 'benchmark_summary.csv'

datasets = ['cifar10', 'cifar100', 'svhn', 'fashionmnist', 'emnist_balanced']
policies = ['head_only', 'conv2_head', 'full_ft']
alpha_map = {'01': 0.1, '03': 0.3, '10': 1.0}

rows = []
for dataset in datasets:
    for path in sorted(RESULTS.glob(f'{dataset}_alpha*_seed*/oracle_results.json')):
        exp = path.parent.name
        alpha_match = re.search(r'alpha(\d+)', exp)
        seed_match = re.search(r'seed(\d+)', exp)
        if alpha_match is None or seed_match is None:
            continue
        alpha_code = alpha_match.group(1)
        if alpha_code not in alpha_map:
            continue
        alpha = alpha_map[alpha_code]
        seed = int(seed_match.group(1))
        with open(path, 'r') as f:
            data = json.load(f)
        fixed = {}
        for p in policies:
            fixed[p] = np.mean([rec[p]['accuracy'] for rec in data.values()]) * 100
        oracle = np.mean([max(rec[p]['accuracy'] for p in policies) for rec in data.values()]) * 100
        best_policy = max(fixed, key=fixed.get)
        rows.append({
            'dataset': dataset,
            'alpha': alpha,
            'seed': seed,
            'head_only': fixed['head_only'],
            'conv2_head': fixed['conv2_head'],
            'full_ft': fixed['full_ft'],
            'best_fixed_policy': best_policy,
            'best_fixed_accuracy': fixed[best_policy],
            'oracle_accuracy': oracle,
            'oracle_headroom': oracle - fixed[best_policy],
        })

df = pd.DataFrame(rows).sort_values(['dataset', 'alpha', 'seed'])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f'Wrote {OUT}')
print(df.head())
