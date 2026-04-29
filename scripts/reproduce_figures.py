from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

sources = [
    ROOT / 'results_FINAL_CAMERA_READY' / 'final_figures',
    ROOT / 'results_FINAL_CAMERA_READY' / 'appendix_figures',
]

copied = 0
for src in sources:
    if not src.exists():
        continue
    for p in src.glob('*.png'):
        shutil.copy2(p, FIG / p.name)
        copied += 1

print(f'Copied {copied} figure files into {FIG}')
