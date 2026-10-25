"""Generate synthetic CXRs with the fine-tuned LoRA-SDXL model (Sec. 3.4).

* DDIM sampler, 200 steps, deterministic (eta = 0)
* 1024 x 1024 output, saved as grayscale PNG
* Optional quality filter: each image is compared with the most similar real
  image of the same class; images with SSIM >= 0.85 are kept, others go to
  `rejected/`. SSIM, PSNR and SNR are written to `quality_metrics.csv`.

Usage (from this folder, GPU recommended):
    python generate_images.py --lora_dir outputs/lora_weights --n_per_class 500 \
        --real_dir /path/to/real_train --output_dir outputs/synthetic
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

from common import config, image_metrics  # noqa: E402
from common.gan_utils import append_csv  # noqa: E402

NEGATIVE_PROMPT = "color, text, watermark, cartoon, illustration, blurry, distorted anatomy, artifacts"


def load_pipeline(base_model: str, vae_model: str | None, lora_dir: str):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype}
    if vae_model:
        kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, **kwargs)
    pipe.load_lora_weights(lora_dir)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def reference_images(real_dir: Path, cls: str, n: int, size: int, seed: int) -> list[np.ndarray]:
    files = [f for f in sorted((real_dir / cls).rglob("*")) if f.suffix.lower() in config.IMAGE_EXTENSIONS]
    random.Random(seed).shuffle(files)
    return [image_metrics.load_gray(f, size) for f in files[:n]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora_dir", required=True, help="Folder containing pytorch_lora_weights.safetensors.")
    p.add_argument("--output_dir", default=str(HERE / "outputs" / "synthetic"))
    p.add_argument("--n_per_class", type=int, default=500,
                   help="Images to KEEP per class (after the SSIM filter when --real_dir is given)")
    p.add_argument("--max_attempts_factor", type=float, default=4.0,
                   help="Stop a class after generating n_per_class x this many images")
    p.add_argument("--classes", nargs="+", default=config.CLASSES)
    p.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--steps", type=int, default=config.LORA["ddim_steps"])
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--resolution", type=int, default=config.LORA["resolution"])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--real_dir", default=None, help="Real images for the SSIM quality filter (optional).")
    p.add_argument("--ssim_threshold", type=float, default=config.LORA["ssim_filter_threshold"])
    p.add_argument("--metric_size", type=int, default=512, help="Resize for metric computation (speed).")
    p.add_argument("--n_reference", type=int, default=100, help="Real reference images per class.")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    import torch

    pipe = load_pipeline(args.base_model, args.vae_model, args.lora_dir)
    out = Path(args.output_dir)
    metrics_csv = out / "quality_metrics.csv"
    out.mkdir(parents=True, exist_ok=True)
    if metrics_csv.exists():
        metrics_csv.unlink()
    summary = {}

    for ci, cls in enumerate(args.classes):
        prompt = config.CLASS_PROMPTS.get(cls, f"a frontal chest X-ray radiograph showing {cls.replace('_', ' ')}")
        refs = (reference_images(Path(args.real_dir), cls, args.n_reference, args.metric_size, args.seed)
                if args.real_dir else [])
        (out / cls).mkdir(parents=True, exist_ok=True)
        if refs:
            (out / "rejected" / cls).mkdir(parents=True, exist_ok=True)
        kept = made = 0
        rows = []
        max_attempts = int(np.ceil(args.n_per_class * args.max_attempts_factor)) if refs else args.n_per_class
        while kept < args.n_per_class and made < max_attempts:
            b = min(args.batch_size, args.n_per_class - kept, max_attempts - made)
            gen = torch.Generator(device=pipe.device).manual_seed(args.seed + ci * 100_000 + made)
            images = pipe(
                prompt=[prompt] * b,
                negative_prompt=[NEGATIVE_PROMPT] * b,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                eta=0.0,  # deterministic DDIM
                generator=gen,
            ).images
            for img in images:
                img = img.convert("L")
                name = f"lora_{cls}_{made:05d}.png"
                accepted, row = True, {"class": cls, "image": name}
                if refs:
                    arr = np.asarray(img.resize((args.metric_size,) * 2), dtype=np.float64) / 255.0
                    m = image_metrics.best_match_metrics(arr, refs)
                    accepted = m["SSIM"] >= args.ssim_threshold
                    row.update({"SSIM": m["SSIM"], "PSNR": m["PSNR"], "SNR": m["SNR"], "accepted": accepted})
                    rows.append(m)
                    append_csv(metrics_csv, row)
                img.save((out / cls if accepted else out / "rejected" / cls) / name)
                kept += accepted
                made += 1
        summary[cls] = {"generated": made, "kept": kept}
        if rows:
            summary[cls]["metrics"] = image_metrics.summarise(rows)
        print(f"  {cls}: generated {made}, kept {kept}"
              + (f", mean SSIM {summary[cls]['metrics']['SSIM']['mean']:.4f}" if rows else ""))
        if kept < args.n_per_class:
            print(f"  warning: only {kept}/{args.n_per_class} {cls} images passed SSIM >= {args.ssim_threshold} "
                  f"after {made} attempts; raise --max_attempts_factor or lower --ssim_threshold")

    (out / "generation_summary.json").write_text(json.dumps({**vars(args), "summary": summary}, indent=2))
    print(f"Synthetic images written to {out}")


if __name__ == "__main__":
    main()
