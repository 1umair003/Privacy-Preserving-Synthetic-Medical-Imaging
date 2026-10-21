"""Compare any synthetic image set with real CXRs: SSIM, PSNR, SNR (+ optional FID).

Works for LoRA-SDXL, SMGAN and SRGAN outputs, so all three generators can be
scored the same way (Table 1). Each synthetic image is matched with the most
similar real image of the same class (highest SSIM).

Usage (from codes/):
    python synthetic_image_generation/image_quality_evaluation/evaluate_quality.py \
        --generated_dir synthetic_image_generation/LoRA_SDXL/outputs/synthetic \
        --real_dir /path/to/real_test --name LoRA_SDXL

    # add --fid to also compute FID (needs: pip install "torchmetrics[image]")
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import config, image_metrics  # noqa: E402


def images_in(folder: Path) -> list[Path]:
    return [f for f in sorted(folder.rglob("*")) if f.suffix.lower() in config.IMAGE_EXTENSIONS]


def compute_fid(gen_files: list[Path], real_files: list[Path], size: int = 299) -> float:
    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048)

    def batches(files, real):
        for i in range(0, len(files), 32):
            arr = [np.asarray(Image.open(f).convert("RGB").resize((size, size))) for f in files[i:i + 32]]
            fid.update(torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2), real=real)

    batches(real_files, True)
    batches(gen_files, False)
    return float(fid.compute())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated_dir", required=True, help="Synthetic images, one sub-folder per class.")
    p.add_argument("--real_dir", required=True, help="Real images, one sub-folder per class.")
    p.add_argument("--name", default="generator", help="Label used in the output file names.")
    p.add_argument("--size", type=int, default=512, help="Images are resized to size x size for metrics.")
    p.add_argument("--n_reference", type=int, default=100, help="Real reference images per class.")
    p.add_argument("--max_per_class", type=int, default=None, help="Limit generated images per class.")
    p.add_argument("--fid", action="store_true", help="Also compute FID (requires torchmetrics[image]).")
    p.add_argument("--output_dir", default=str(HERE / "results"))
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    gen_root, real_root, out = Path(args.generated_dir), Path(args.real_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    classes = sorted(d.name for d in gen_root.iterdir() if d.is_dir() and (real_root / d.name).is_dir())
    if not classes:
        sys.exit("No matching class folders between --generated_dir and --real_dir.")

    rows, all_gen, all_real = [], [], []
    for cls in classes:
        real_files = images_in(real_root / cls)
        rng.shuffle(real_files)
        refs = [image_metrics.load_gray(f, args.size) for f in real_files[:args.n_reference]]
        gen_files = images_in(gen_root / cls)[:args.max_per_class]
        all_gen += gen_files
        all_real += real_files[:args.n_reference]
        for f in gen_files:
            m = image_metrics.best_match_metrics(image_metrics.load_gray(f, args.size), refs)
            rows.append({"class": cls, "image": f.name, "SSIM": m["SSIM"], "PSNR": m["PSNR"], "SNR": m["SNR"],
                         "matched_real": real_files[m["reference_index"]].name})
        print(f"  {cls}: {len(gen_files)} synthetic vs {len(refs)} real references")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{args.name}_per_image.csv", index=False)
    finite = df.replace([np.inf, -np.inf], np.nan)
    per_class = finite.groupby("class")[["SSIM", "PSNR", "SNR"]].agg(["mean", "std"])
    per_class.to_csv(out / f"{args.name}_per_class.csv")

    summary = {
        "generator": args.name,
        "n_images": int(len(df)),
        "SSIM": float(finite["SSIM"].mean()),
        "PSNR_dB": float(finite["PSNR"].mean()),
        "SNR_dB": float(finite["SNR"].mean()),
    }
    if args.fid:
        summary["FID"] = compute_fid(all_gen, all_real)
    (out / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPer-class (mean, std):")
    print(per_class.round(4).to_string())
    print("\nOverall:", json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
