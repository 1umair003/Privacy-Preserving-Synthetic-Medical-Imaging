"""SMGAN (style-modulated GAN) baseline for class-conditional CXR synthesis (Sec. 3.3, Eqs. 2-4).

Generator (StyleGAN-style):
    z + class embedding -> mapping network f (8 FC layers) -> w
    learned 4x4 constant -> [upsample, conv, noise, AdaIN(w)] x N -> grayscale image
    AdaIN(x_i, y) = y_s,i * (x_i - mu(x_i)) / sigma(x_i) + y_b,i,   y = A(w)        (Eq. 3)

Losses:
    Discriminator: WGAN-GP (Eq. 2, lambda = 10) + lazy R1 penalty on real images
    Generator:     L_G = L_adv + lambda_PL * L_path                                 (Eq. 4)

Every `--eval_every` epochs, generated images are scored against the most similar
real image of the same class with SSIM, PSNR and SNR.

Usage (from this folder):
    # train
    python smgan.py train --data_dir /path/to/real_train --epochs 200

    # generate synthetic images per class with a trained generator
    python smgan.py generate --checkpoint outputs/checkpoints/generator_last.pt \
        --n_per_class 500 --output_dir outputs/generated
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import config, image_metrics  # noqa: E402
from common.gan_utils import (  # noqa: E402
    GrayscaleCXRDataset, append_csv, get_device, list_class_images, save_grid, save_png,
    seed_everything, to_unit,
)


# ============================================================ generator
class MappingNetwork(nn.Module):
    """f: Z -> W with 8 fully connected layers; class label is embedded and concatenated to z."""

    def __init__(self, z_dim: int, w_dim: int, num_classes: int, n_layers: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_classes, z_dim)
        layers, d = [], z_dim * 2
        for _ in range(n_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        z = F.normalize(z, dim=1)
        return self.net(torch.cat([z, F.normalize(self.embed(c), dim=1)], dim=1))


class AdaIN(nn.Module):
    """Adaptive instance normalisation with style y = A(w) (Eq. 3)."""

    def __init__(self, ch: int, w_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.affine = nn.Linear(w_dim, ch * 2)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:ch] = 1.0   # y_s starts at 1
            self.affine.bias[ch:] = 0.0   # y_b starts at 0

    def forward(self, x, w):
        y_s, y_b = self.affine(w).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return y_s * self.norm(x) + y_b


class NoiseInjection(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        return x + self.weight * torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)


class StyleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(out_ch, w_dim)

    def forward(self, x, w):
        return self.adain(F.leaky_relu(self.noise(self.conv(x)), 0.2), w)


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.l1 = StyleLayer(in_ch, out_ch, w_dim)
        self.l2 = StyleLayer(out_ch, out_ch, w_dim)

    def forward(self, x, w):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.l2(self.l1(x, w), w)


def _channels(res: int, base: int = 512, min_ch: int = 32) -> list[int]:
    n_blocks = int(math.log2(res)) - 2  # 4x4 -> res
    return [max(base // (2 ** max(i - 2, 0)), min_ch) for i in range(n_blocks + 1)]


class Generator(nn.Module):
    def __init__(self, res: int, num_classes: int, z_dim: int = 512, w_dim: int = 512, base_ch: int = 512):
        super().__init__()
        if res < 8 or res & (res - 1):
            raise ValueError("--resolution must be a power of two >= 8")
        self.z_dim = z_dim
        chs = _channels(res, base_ch)
        self.mapping = MappingNetwork(z_dim, w_dim, num_classes)
        self.const = nn.Parameter(torch.randn(1, chs[0], 4, 4))
        self.initial = StyleLayer(chs[0], chs[0], w_dim)
        self.blocks = nn.ModuleList(SynthesisBlock(chs[i], chs[i + 1], w_dim) for i in range(len(chs) - 1))
        self.to_gray = nn.Conv2d(chs[-1], 1, 1)

    def synthesis(self, w):
        x = self.initial(self.const.expand(w.size(0), -1, -1, -1), w)
        for block in self.blocks:
            x = block(x, w)
        return torch.tanh(self.to_gray(x))

    def forward(self, z, c, return_w: bool = False):
        w = self.mapping(z, c)
        img = self.synthesis(w)
        return (img, w) if return_w else img


# ============================================================ discriminator
class Discriminator(nn.Module):
    """Convolutional critic with projection class conditioning (no BatchNorm, for WGAN-GP)."""

    def __init__(self, res: int, num_classes: int, base_ch: int = 512):
        super().__init__()
        chs = list(reversed(_channels(res, base_ch)))
        self.from_gray = nn.Sequential(nn.Conv2d(1, chs[0], 1), nn.LeakyReLU(0.2))
        blocks = []
        for i in range(len(chs) - 1):
            blocks += [nn.Conv2d(chs[i], chs[i], 3, padding=1), nn.LeakyReLU(0.2),
                       nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.LeakyReLU(0.2), nn.AvgPool2d(2)]
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Sequential(nn.Conv2d(chs[-1], chs[-1], 3, padding=1), nn.LeakyReLU(0.2),
                                   nn.Flatten(), nn.Linear(chs[-1] * 16, chs[-1]), nn.LeakyReLU(0.2))
        self.out = nn.Linear(chs[-1], 1)
        self.embed = nn.Embedding(num_classes, chs[-1])

    def forward(self, x, c):
        h = self.final(self.blocks(self.from_gray(x)))
        return self.out(h).squeeze(1) + (self.embed(c) * h).sum(1)


# ============================================================ regularisers
def gradient_penalty(D, real, fake, c):
    """WGAN-GP term: E[(||grad D(x_hat)||_2 - 1)^2], x_hat = eps * x_r + (1 - eps) * G(z)."""
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad = torch.autograd.grad(D(x_hat, c).sum(), x_hat, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def r1_penalty(D, real, c):
    real = real.detach().requires_grad_(True)
    grad = torch.autograd.grad(D(real, c).sum(), real, create_graph=True)[0]
    return grad.pow(2).flatten(1).sum(1).mean()


def path_length_penalty(img, w, pl_mean, decay: float = 0.01):
    """Path-length regularisation: penalise deviation of ||J_w^T y|| from its running mean."""
    noise = torch.randn_like(img) / math.sqrt(img.size(2) * img.size(3))
    grad = torch.autograd.grad((img * noise).sum(), w, create_graph=True)[0]
    lengths = (grad.pow(2).sum(1) + 1e-8).sqrt()  # epsilon avoids an infinite gradient at zero
    new_mean = pl_mean + decay * (lengths.mean().detach() - pl_mean)
    return (lengths - new_mean).pow(2).mean(), new_mean


# ============================================================ evaluation
@torch.no_grad()
def evaluate(G, refs_by_class, classes, n_per_class, device):
    """SSIM / PSNR / SNR of generated images vs the most similar real image of the same class."""
    G.eval()
    rows, per_class = [], {}
    for ci, name in enumerate(classes):
        refs = refs_by_class.get(ci, [])
        if not refs:
            continue
        z = torch.randn(n_per_class, G.z_dim, device=device)
        imgs = G(z, torch.full((n_per_class,), ci, device=device, dtype=torch.long))
        class_rows = [image_metrics.best_match_metrics(to_unit(x), refs) for x in imgs]
        per_class[name] = image_metrics.summarise(class_rows)
        rows += class_rows
    G.train()
    return image_metrics.summarise(rows), per_class


# ============================================================ training
def train(args):
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.output_dir)
    for sub in ("checkpoints", "samples"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    paths, labels, classes = list_class_images(args.data_dir)
    ds = GrayscaleCXRDataset(paths, labels, args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    drop_last=len(ds) > args.batch_size)

    rng = random.Random(args.seed)
    refs_by_class = {}
    for ci in range(len(classes)):
        idx = [i for i, l in enumerate(labels) if l == ci]
        refs_by_class[ci] = [image_metrics.load_gray(paths[i], args.resolution)
                             for i in rng.sample(idx, min(args.n_reference, len(idx)))]

    G = Generator(args.resolution, len(classes), args.z_dim, args.z_dim, args.base_channels).to(device)
    D = Discriminator(args.resolution, len(classes), args.base_channels).to(device)
    opt_g = torch.optim.Adam([
        {"params": G.mapping.parameters(), "lr": args.lr * 0.01},  # slower mapping network
        {"params": [p for n, p in G.named_parameters() if not n.startswith("mapping.")]},
    ], lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99))
    print(f"Device: {device} | {len(ds)} images | classes {classes} | resolution {args.resolution}")

    fixed_z = torch.randn(len(classes) * 4, args.z_dim, device=device)
    fixed_c = torch.arange(len(classes), device=device).repeat_interleave(4)
    pl_mean = torch.zeros((), device=device)
    history_csv, step = out / "metrics_per_epoch.csv", 0
    if history_csv.exists():
        history_csv.unlink()

    for epoch in range(1, args.epochs + 1):
        g_tot = d_tot = n = 0
        for real, c in dl:
            real, c = real.to(device), c.to(device)
            b = real.size(0)

            # ---- critic: WGAN-GP (Eq. 2) + lazy R1
            for _ in range(args.n_critic):
                z = torch.randn(b, args.z_dim, device=device)
                fake = G(z, c).detach()
                d_loss = D(fake, c).mean() - D(real, c).mean() + args.lambda_gp * gradient_penalty(D, real, fake, c)
                if args.r1_every and step % args.r1_every == 0:
                    d_loss = d_loss + args.lambda_r1 * 0.5 * r1_penalty(D, real, c) * args.r1_every
                d_loss = d_loss + 1e-3 * D(real, c).pow(2).mean()  # drift term keeps critic outputs bounded
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # ---- generator: L_adv + lambda_PL * L_path (Eq. 4)
            z = torch.randn(b, args.z_dim, device=device)
            img, w = G(z, c, return_w=True)
            g_loss = -D(img, c).mean()
            if args.pl_every and step % args.pl_every == 0:
                pl, pl_mean = path_length_penalty(img, w, pl_mean)
                g_loss = g_loss + args.lambda_pl * pl * args.pl_every
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_tot += g_loss.item()
            d_tot += d_loss.item()
            n += 1
            step += 1

        row = {"epoch": epoch, "g_loss": g_tot / max(n, 1), "d_loss": d_tot / max(n, 1)}
        msg = f"Epoch {epoch:03d}/{args.epochs} | G {row['g_loss']:.4f} D {row['d_loss']:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            overall, per_class = evaluate(G, refs_by_class, classes, args.n_eval_per_class, device)
            row.update({k: overall[k]["mean"] for k in ("SSIM", "PSNR", "SNR")})
            msg += f" | SSIM {row['SSIM']:.4f} PSNR {row['PSNR']:.2f} dB SNR {row['SNR']:.2f} dB"
            (out / "latest_quality_metrics.json").write_text(
                json.dumps({"epoch": epoch, "overall": overall, "per_class": per_class}, indent=2))
            with torch.no_grad():
                G.eval()
                save_grid(G(fixed_z, fixed_c), out / "samples" / f"epoch_{epoch:03d}.png", nrow=4)
                G.train()
        append_csv(history_csv, {k: row.get(k, "") for k in ("epoch", "g_loss", "d_loss", "SSIM", "PSNR", "SNR")})
        print(msg)
        torch.save({"G": G.state_dict(), "classes": classes, "resolution": args.resolution,
                    "z_dim": args.z_dim, "base_channels": args.base_channels},
                   out / "checkpoints" / "generator_last.pt")
        torch.save(D.state_dict(), out / "checkpoints" / "discriminator_last.pt")

    (out / "config.json").write_text(json.dumps({**vars(args), "classes": classes}, indent=2, default=str))
    print(f"Done. Outputs in {out}")


# ============================================================ generation
@torch.no_grad()
def generate(args):
    seed_everything(args.seed)
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    G = Generator(ckpt["resolution"], len(classes), ckpt["z_dim"], ckpt["z_dim"], ckpt["base_channels"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    out = Path(args.output_dir)
    for ci, name in enumerate(classes):
        (out / name).mkdir(parents=True, exist_ok=True)
        done = 0
        while done < args.n_per_class:
            b = min(args.batch_size, args.n_per_class - done)
            imgs = G(torch.randn(b, ckpt["z_dim"], device=device),
                     torch.full((b,), ci, device=device, dtype=torch.long))
            for x in imgs:
                save_png(x, out / name / f"smgan_{name}_{done:05d}.png")
                done += 1
        print(f"  {name}: {done} images")
    print(f"Generated images written to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train SMGAN and report SSIM / PSNR / SNR.")
    t.add_argument("--data_dir", required=True, help="Real CXR folder, one sub-folder per class.")
    t.add_argument("--output_dir", default=str(HERE / "outputs"))
    t.add_argument("--resolution", type=int, default=256, help="Power of two; paper's GAN baseline used 512.")
    t.add_argument("--z_dim", type=int, default=512)
    t.add_argument("--base_channels", type=int, default=512)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--n_critic", type=int, default=1)
    t.add_argument("--lambda_gp", type=float, default=10.0, help="WGAN-GP coefficient (paper: 10).")
    t.add_argument("--lambda_r1", type=float, default=10.0)
    t.add_argument("--r1_every", type=int, default=16, help="Lazy R1 interval (0 disables).")
    t.add_argument("--lambda_pl", type=float, default=2.0)
    t.add_argument("--pl_every", type=int, default=4, help="Lazy path-length interval (0 disables).")
    t.add_argument("--eval_every", type=int, default=5)
    t.add_argument("--n_eval_per_class", type=int, default=16)
    t.add_argument("--n_reference", type=int, default=64, help="Real reference images per class for metrics.")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=config.SEED)

    g = sub.add_parser("generate", help="Generate synthetic images per class.")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--n_per_class", type=int, default=500)
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--output_dir", default=str(HERE / "outputs" / "generated"))
    g.add_argument("--seed", type=int, default=config.SEED)

    args = p.parse_args()
    train(args) if args.command == "train" else generate(args)


if __name__ == "__main__":
    main()
