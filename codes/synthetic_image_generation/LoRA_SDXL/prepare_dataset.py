"""Prepare real CXRs for LoRA-SDXL fine-tuning (Sec. 3.2 and 3.4).

Creates a Hugging Face `imagefolder` dataset:

    output_dir/
        metadata.jsonl      {"file_name": "...", "text": "<class prompt>"}
        <images>.png        RGB, resized to --resolution

Tuberculosis and COVID-19 are oversampled 2x (paper: class-conditional
oversampling of the two most under-represented classes).

Usage (from this folder):
    python prepare_dataset.py --real_dir /path/to/real_train --output_dir data/lora_train
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from PIL import Image  # noqa: E402

from common import config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real_dir", required=True, help="Real CXR folder, one sub-folder per class.")
    p.add_argument("--output_dir", default=str(HERE / "data" / "lora_train"))
    p.add_argument("--resolution", type=int, default=config.LORA["resolution"])
    p.add_argument("--oversample_factor", type=int, default=config.LORA["oversample_factor"])
    args = p.parse_args()

    real, out = Path(args.real_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for class_dir in sorted(d for d in real.iterdir() if d.is_dir()):
        name = class_dir.name
        prompt = config.CLASS_PROMPTS.get(name)
        if prompt is None:
            readable = name.replace("_", " ").lower()
            prompt = f"a frontal chest X-ray radiograph showing {readable}, high resolution, grayscale"
            print(f"  note: no preset prompt for '{name}', using: {prompt}")
        copies = args.oversample_factor if name in config.LORA["oversample_classes"] else 1
        files = [f for f in sorted(class_dir.rglob("*")) if f.suffix.lower() in config.IMAGE_EXTENSIONS]
        for f in files:
            img = Image.open(f).convert("RGB").resize((args.resolution, args.resolution), Image.BICUBIC)
            for k in range(copies):
                fname = f"{name}__{f.stem}__{k}.png"
                if k == 0:
                    img.save(out / fname)
                else:
                    (out / fname).write_bytes((out / f"{name}__{f.stem}__0.png").read_bytes())
                records.append({"file_name": fname, "text": prompt})
        print(f"  {name}: {len(files)} images x{copies} = {len(files) * copies}")

    with open(out / "metadata.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} training records to {out}")


if __name__ == "__main__":
    main()
