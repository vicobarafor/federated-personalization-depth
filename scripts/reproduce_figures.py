from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'

required = [
    'winner_heatmap.png',
    'oracle_headroom.png',
    'client_depth_distribution.png',
    'client_oracle_gain_histogram.png',
    'oracle_fixed_disagreement_matrix.png',
    'oracle_win_margin_histogram.png',
]

for name in required:
    path = FIG / name
    print(f'{name}:', 'OK' if path.exists() else 'MISSING')