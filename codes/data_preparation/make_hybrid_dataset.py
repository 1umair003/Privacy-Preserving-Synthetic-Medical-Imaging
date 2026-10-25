"""Build the hybrid training set: 50% real + 50% synthetic images per class (Group 1).

Usage (from codes/):
    python data_preparation/make_hybrid_dataset.py \
        --real_dir /path/to/real_train \
        --synthetic_dir /path/to/synthetic_train \
        --output_dir /path/to/hybrid_train

For each class the script takes N = min(#real, #synthetic) images from each
source (or --per_source if given) and copies them into output_dir/<class>/
with `real_` / `synth_` prefixes.
"""

import argparse
import random
import shutil
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(folder: Path) -> list[Path]:
    return sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXTS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real_dir", required=True)
    p.add_argument("--synthetic_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_source", type=int, default=None,
                   help="Images per class from each source (default: balanced maximum).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    real_root, synth_root, out_root = map(Path, (args.real_dir, args.synthetic_dir, args.output_dir))
    classes = sorted(d.name for d in real_root.iterdir() if d.is_dir())

    for c in classes:
        real, synth = images(real_root / c), images(synth_root / c)
        if not real or not synth:
            print(f"  {c}: skipped (real={len(real)}, synthetic={len(synth)})")
            continue
        n = min(len(real), len(synth))
        if args.per_source:
            n = min(n, args.per_source)
        out = out_root / c
        out.mkdir(parents=True, exist_ok=True)
        for tag, files in (("real", real), ("synth", synth)):
            for f in rng.sample(files, n):
                shutil.copy2(f, out / f"{tag}_{f.name}")
        print(f"  {c}: {n} real + {n} synthetic")
    print(f"Hybrid dataset written to {out_root}")


if __name__ == "__main__":
    main()
