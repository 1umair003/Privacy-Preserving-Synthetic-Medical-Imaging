"""SRGAN baseline for chest X-ray synthesis (Sec. 3.3, Eqs. 5-7).

The generator super-resolves low-resolution CXRs (bicubic down-sampled by
`--scale`) back to `--hr_size`. It is trained with a perceptual loss:

    L_perceptual = L_content (VGG19 feature MSE, Eq. 6) + lambda_adv * L_adv (Eq. 7)

After every epoch the model is evaluated on a held-out split with
SSIM, PSNR and SNR against the real high-resolution images, alongside a
bicubic-upsampling reference.

Usage (from this folder):
    # train
    python srgan.py train --data_dir /path/to/real_train --epochs 100

    # super-resolve a folder of images with a trained generator
    python srgan.py infer --checkpoint outputs/checkpoints/generator_best.pt \
        --input_dir /path/to/low_res --output_dir outputs/generated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import config, image_metrics  # noqa: E402
from common.gan_utils import (  # noqa: E402
    GrayscaleCXRDataset, append_csv, get_device, list_class_images, save_grid, save_png,
    seed_everything, to_unit,
)


# ============================================================ networks
class ResidualBlock(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class UpsampleBlock(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU())

    def forward(self, x):
        return self.body(x)


class Generator(nn.Module):
    """SRResNet generator: 16 residual blocks + sub-pixel up-sampling (x2 per block)."""

    def __init__(self, scale: int = 4, n_res: int = 16, ch: int = 64, in_ch: int = 1):
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("scale must be 2, 4 or 8")
        self.head = nn.Sequential(nn.Conv2d(in_ch, ch, 9, padding=4), nn.PReLU())
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(n_res)])
        self.mid = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        self.up = nn.Sequential(*[UpsampleBlock(ch) for _ in range(int(np.log2(scale)))])
        self.tail = nn.Conv2d(ch, in_ch, 9, padding=4)

    def forward(self, x):
        h = self.head(x)
        h = h + self.mid(self.res(h))
        return torch.tanh(self.tail(self.up(h)))


class Discriminator(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        layers, ch_in = [], in_ch
        for i, (ch, stride) in enumerate([(base, 1), (base, 2), (base * 2, 1), (base * 2, 2),
                                          (base * 4, 1), (base * 4, 2), (base * 8, 1), (base * 8, 2)]):
            layers.append(nn.Conv2d(ch_in, ch, 3, stride=stride, padding=1))
            if i > 0:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch_in = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 8, 1024), nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))  # logits


class VGGContentLoss(nn.Module):
    """MSE between VGG19 feature maps (phi_5,4) of SR and HR images (Eq. 6)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(vgg.features.children())[:36]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = ((x + 1) / 2).repeat(1, 3, 1, 1)  # grayscale [-1,1] -> RGB [0,1]
        return (x - self.mean) / self.std

    def forward(self, sr, hr):
        return F.mse_loss(self.features(self._prep(sr)), self.features(self._prep(hr)))


# ============================================================ helpers
def downsample(hr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False,
                         antialias=True).clamp(-1, 1)


@torch.no_grad()
def evaluate(G: nn.Module, loader: DataLoader, scale: int, device: torch.device) -> dict:
    G.eval()
    sr_rows, bicubic_rows = [], []
    for hr, _ in loader:
        hr = hr.to(device)
        lr = downsample(hr, scale)
        sr = G(lr)
        bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1)
        for i in range(hr.size(0)):
            ref = to_unit(hr[i])
            sr_rows.append(image_metrics.all_metrics(ref, to_unit(sr[i])))
            bicubic_rows.append(image_metrics.all_metrics(ref, to_unit(bic[i])))
    G.train()
    return {"srgan": image_metrics.summarise(sr_rows), "bicubic": image_metrics.summarise(bicubic_rows)}


# ============================================================ training
def train(args):
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    paths, labels, classes = list_class_images(args.data_dir)
    tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=args.val_fraction,
                                              stratify=labels, random_state=args.seed)
    train_dl = DataLoader(GrayscaleCXRDataset(tr_p, tr_l, args.hr_size), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.workers, drop_last=len(tr_p) > args.batch_size)
    val_dl = DataLoader(GrayscaleCXRDataset(va_p, va_l, args.hr_size, hflip=False),
                        batch_size=args.batch_size, num_workers=args.workers)
    print(f"Device: {device} | train {len(tr_p)} | val {len(va_p)} | classes {classes}")

    G = Generator(args.scale, args.n_res_blocks).to(device)
    D = Discriminator().to(device)
    content_loss = VGGContentLoss(pretrained=not args.no_pretrained_vgg).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    best_ssim, history_csv = -1.0, out / "metrics_per_epoch.csv"
    if history_csv.exists():
        history_csv.unlink()

    for epoch in range(1, args.epochs + 1):
        g_tot = d_tot = n = 0
        for hr, _ in train_dl:
            hr = hr.to(device)
            lr = downsample(hr, args.scale)
            sr = G(lr)

            # ---- discriminator
            real_logits, fake_logits = D(hr), D(sr.detach())
            d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ---- generator: content + adversarial (+ optional pixel MSE warm start)
            fake_logits = D(sr)
            adv = bce(fake_logits, torch.ones_like(fake_logits))  # = -E[log D(G(I_LR))]
            g_loss = content_loss(sr, hr) + args.lambda_adv * adv
            if args.lambda_pixel > 0:
                g_loss = g_loss + args.lambda_pixel * F.mse_loss(sr, hr)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_tot += g_loss.item()
            d_tot += d_loss.item()
            n += 1

        m = evaluate(G, val_dl, args.scale, device)
        row = {
            "epoch": epoch, "g_loss": g_tot / max(n, 1), "d_loss": d_tot / max(n, 1),
            "SSIM": m["srgan"]["SSIM"]["mean"], "PSNR": m["srgan"]["PSNR"]["mean"], "SNR": m["srgan"]["SNR"]["mean"],
            "bicubic_SSIM": m["bicubic"]["SSIM"]["mean"], "bicubic_PSNR": m["bicubic"]["PSNR"]["mean"],
            "bicubic_SNR": m["bicubic"]["SNR"]["mean"],
        }
        append_csv(history_csv, row)
        print(f"Epoch {epoch:03d}/{args.epochs} | G {row['g_loss']:.4f} D {row['d_loss']:.4f} | "
              f"SSIM {row['SSIM']:.4f} PSNR {row['PSNR']:.2f} dB SNR {row['SNR']:.2f} dB "
              f"(bicubic SSIM {row['bicubic_SSIM']:.4f})")

        torch.save(G.state_dict(), out / "checkpoints" / "generator_last.pt")
        torch.save(D.state_dict(), out / "checkpoints" / "discriminator_last.pt")
        if row["SSIM"] > best_ssim:
            best_ssim = row["SSIM"]
            torch.save(G.state_dict(), out / "checkpoints" / "generator_best.pt")
            (out / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **m}, indent=2))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                hr, _ = next(iter(val_dl))
                hr = hr[:4].to(device)
                lr = downsample(hr, args.scale)
                bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
                G.eval()
                grid = torch.cat([bic, G(lr), hr])
                G.train()
            save_grid(grid, out / "samples" / f"epoch_{epoch:03d}.png", nrow=hr.size(0))

    (out / "config.json").write_text(json.dumps({**vars(args), "classes": classes}, indent=2, default=str))
    print(f"Done. Best validation SSIM {best_ssim:.4f}. Outputs in {out}")


# ============================================================ inference
@torch.no_grad()
def infer(args):
    device = get_device()
    G = Generator(args.scale, args.n_res_blocks).to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    files = [f for f in sorted(in_dir.rglob("*")) if f.suffix.lower() in config.IMAGE_EXTENSIONS]
    for f in files:
        img = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 127.5 - 1.0
        lr = torch.from_numpy(img)[None, None].to(device)
        target = out_dir / f.relative_to(in_dir).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        save_png(G(lr)[0], target)
    print(f"Super-resolved {len(files)} images into {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train SRGAN and report SSIM / PSNR / SNR each epoch.")
    t.add_argument("--data_dir", required=True, help="Real CXR folder, one sub-folder per class.")
    t.add_argument("--output_dir", default=str(HERE / "outputs"))
    t.add_argument("--hr_size", type=int, default=512, help="High-resolution size (paper: 512).")
    t.add_argument("--scale", type=int, default=4)
    t.add_argument("--n_res_blocks", type=int, default=16)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lambda_adv", type=float, default=1e-3)
    t.add_argument("--lambda_pixel", type=float, default=0.0, help="Optional extra pixel-wise MSE weight.")
    t.add_argument("--val_fraction", type=float, default=0.1)
    t.add_argument("--sample_every", type=int, default=5)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=config.SEED)
    t.add_argument("--no_pretrained_vgg", action="store_true",
                   help="Use a randomly initialised VGG19 (offline testing only).")

    i = sub.add_parser("infer", help="Super-resolve a folder of low-resolution images.")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--input_dir", required=True)
    i.add_argument("--output_dir", default=str(HERE / "outputs" / "generated"))
    i.add_argument("--scale", type=int, default=4)
    i.add_argument("--n_res_blocks", type=int, default=16)

    args = p.parse_args()
    train(args) if args.command == "train" else infer(args)


if __name__ == "__main__":
    main()
