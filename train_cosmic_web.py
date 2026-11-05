#!/usr/bin/env python
"""Train a macrocanonical generator on the Quijote cosmic-web benchmark.

The generator y = x + U(x) (a circular-padded U-Net) maps Gaussian random
fields with the target's power spectrum to fields whose WPH statistics
match the mean over 100 Quijote density slices.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pywph as pw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

# Activation checkpointing in the U-Net body, to fit batch size 200 in memory.
USE_CHECKPOINTING = True


def radial_profile(data):
    center = data.shape[0] / 2
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype("int32")
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr


def compute_spectrum_map(power1d, size):
    power_map = np.zeros((size, size), dtype=float)
    k_map = np.zeros((size, size), dtype=float)
    for (i, j), _ in np.ndenumerate(power_map):
        k1 = i - size / 2.0
        k2 = j - size / 2.0
        k_map[i, j] = np.sqrt(k1 * k1 + k2 * k2)
        if k_map[i, j] == 0:
            power_map[i, j] = 1e-15
        else:
            power_map[i, j] = power1d[int(k_map[i, j])]
    return power_map


def gaussian_mock(spectrum_map_flat_sqrt, image_size):
    gaussian_field = (
        torch.normal(0, spectrum_map_flat_sqrt)
        + 1j * torch.normal(0, spectrum_map_flat_sqrt)
    )
    gaussian_field = torch.fft.ifft2(
        torch.fft.fftshift(gaussian_field.reshape((image_size, image_size)))
    ).imag
    return gaussian_field


def gaussian_mock_batch(batch_size, spectrum_map, image_size, device):
    spectrum_map_flat_sqrt = torch.sqrt(spectrum_map.flatten())
    gaussian_fields = torch.empty(
        (batch_size, image_size, image_size), dtype=torch.float32, device=device
    )
    for i in range(batch_size):
        gaussian_fields[i] = gaussian_mock(spectrum_map_flat_sqrt, image_size)
    return gaussian_fields


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False,
                padding_mode="circular",
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False,
                padding_mode="circular",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CircularUpsample2d(nn.Module):
    """Bilinear upsampling with periodic boundaries (circular pad, interpolate, crop)."""

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = int(scale_factor)

    def forward(self, x):
        s = self.scale_factor
        x = F.pad(x, [1, 1, 1, 1], mode="circular")
        x = F.interpolate(
            x, scale_factor=s, mode="bilinear", align_corners=False
        )
        return x[..., s:-s, s:-s]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = CircularUpsample2d(scale_factor=2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            mode="circular" if (diffX or diffY) else "constant",
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_checkpointing=USE_CHECKPOINTING):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.use_checkpointing = use_checkpointing

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self._maybe_ckpt(self.down1, x1)
        x3 = self._maybe_ckpt(self.down2, x2)
        x4 = self._maybe_ckpt(self.down3, x3)
        x5 = self._maybe_ckpt(self.down4, x4)
        x = self._maybe_ckpt(self.up1, x5, x4)
        x = self._maybe_ckpt(self.up2, x, x3)
        x = self._maybe_ckpt(self.up3, x, x2)
        x = self._maybe_ckpt(self.up4, x, x1)
        x = self.outc(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 1e-4)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 1)
        nn.init.constant_(m.bias.data, 0)


def load_data(path, M, device_wph, device_net):
    """Return the preprocessed target slices (on ``device_wph``) and the 2-D
    power spectrum for the Gaussian input (on ``device_net``)."""
    data = np.log10(np.load(path).real.astype(np.float32)[:100] + 1)
    pool = data.shape[1] // M
    data = data.reshape(data.shape[0], M, pool, M, pool).mean(axis=(2, 4))

    data = (data - data.mean()) / data.std() / 8

    signal_power = []
    for i in range(data.shape[0]):
        f = np.fft.fftshift(np.fft.fft2(data[i] - np.mean(data[i])))
        signal_power.append(radial_profile(np.real(f * np.conj(f))))
    signal_power = np.mean(np.array(signal_power), axis=0)

    data_torch = torch.from_numpy(data).to(device_wph)
    spectrum_map = torch.from_numpy(
        compute_spectrum_map(signal_power, M)
    ).to(device_net)
    return data_torch, spectrum_map


def train(
    netG,
    wph_op,
    coeffs,
    spectrum_map,
    image_size,
    norm,
    *,
    num_iters,
    batch_size,
    optimizer,
    scheduler=None,
    n_chunk=1,
    log_every=25,
    device_net,
    device_wph,
):
    """Train ``netG`` to match the target WPH coefficients ``coeffs`` in
    expectation. Returns the per-iteration losses."""
    chunk_indices = list(np.linspace(0, batch_size, n_chunk + 1).astype(int))
    losses = []

    print(
        f"Training: {num_iters} iterations, batch_size={batch_size}, "
        f"n_chunk={n_chunk}, net={device_net}, wph={device_wph}",
        flush=True,
    )

    t0 = time.time()
    for iteration in range(num_iters):
        optimizer.zero_grad()

        noise = gaussian_mock_batch(batch_size, spectrum_map, image_size, device_net)
        noise = noise.reshape(batch_size, 1, image_size, image_size)
        fake = netG(noise) + noise
        fake = fake.reshape(batch_size, image_size, image_size)

        loss_tot = torch.zeros(1)
        for i in range(n_chunk):
            chunk = fake[chunk_indices[i] : chunk_indices[i + 1]].to(device_wph)
            coeffs_step = wph_op.apply(chunk, norm=norm, pbc=True)
            loss = (
                torch.mean(torch.abs(coeffs_step.mean(axis=0) - coeffs) ** 2.0)
                * 1e3
                / batch_size
            )
            loss.backward(retain_graph=i < n_chunk - 1)
            loss_tot += torch.clone(loss).detach().cpu()
            del coeffs_step, loss, chunk

        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss_tot.item())

        if iteration % log_every == 0:
            print(
                f"iter {iteration}, loss={float(loss_tot):.4f}, "
                f"time={time.time() - t0:.1f}s, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            t0 = time.time()

    return losses


def make_sample_comparison(
    netG,
    target_images,
    spectrum_map,
    image_size,
    save_path,
    n_samples=4,
    vmin=-0.4,
    vmax=0.4,
    bn_mode="eval",
):
    """Save a figure of target slices, Gaussian inputs and generated samples.

    ``bn_mode="eval"`` uses BatchNorm running stats, so samples are
    independent; ``"train"`` uses batch stats and is for debugging only."""
    if bn_mode not in ("train", "eval"):
        raise ValueError(f"bn_mode must be 'train' or 'eval', got {bn_mode!r}")

    device = next(netG.parameters()).device
    was_training = netG.training
    if bn_mode == "eval":
        netG.eval()
    else:
        netG.train()
    try:
        with torch.no_grad():
            noise = gaussian_mock_batch(n_samples, spectrum_map, image_size, device)
            noise_in = noise.reshape(n_samples, 1, image_size, image_size)
            fake = (netG(noise_in) + noise_in).reshape(n_samples, image_size, image_size)
    finally:
        netG.train(was_training)

    targets_np = target_images[:n_samples].detach().cpu().numpy()
    noise_np = noise.detach().cpu().numpy()
    fake_np = fake.detach().cpu().numpy()

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "mathtext.fontset": "cm",
        "savefig.dpi": 300,
    }
    rows = [
        (r"Quijote target", targets_np),
        (r"Gaussian input $x$", noise_np),
        (r"Neural synthesis $F_\theta(x)$", fake_np),
    ]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            len(rows),
            n_samples,
            figsize=(n_samples * 1.5 + 0.6, len(rows) * 1.5 + 0.2),
            constrained_layout=True,
        )
        for r, (label, batch) in enumerate(rows):
            for c in range(n_samples):
                ax = axes[r, c]
                im = ax.imshow(
                    batch[c],
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 0:
                    ax.set_ylabel(label)

        cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.015)
        cbar.set_label(r"normalised density")
        cbar.outline.set_visible(False)
        fig.text(
            0.005,
            0.005,
            f"BN: {bn_mode}",
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.4",
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_devices(args):
    """Resolve the (net, wph) ``torch.device`` pair from CLI args."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")
    n_gpus = torch.cuda.device_count()
    net_idx, wph_idx = args.gpus
    for label, idx in (("net", net_idx), ("wph", wph_idx)):
        if idx < 0 or idx >= n_gpus:
            raise RuntimeError(
                f"Requested {label} device cuda:{idx}, but only {n_gpus} "
                f"GPU(s) are visible. Pick a valid index via --gpus."
            )
    return torch.device(f"cuda:{net_idx}"), torch.device(f"cuda:{wph_idx}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", nargs=2, type=int, default=[0, 1], metavar=("NET", "WPH"),
        help="GPU indices for the U-Net and the pywph operator. Default: 0 1.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed."
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    M = 128
    J = 5
    L = 4
    dn = 0
    dj = None
    dl = 2
    image_size = M
    norm = "auto"

    run_id = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"cosmic_web_{run_id}"

    data_path = "./data/Quijote_fidu_300.npy"
    save_path = f"{prefix}_netG"
    spectrum_path = f"{prefix}_spectrum.pt"
    loss_plot_path = f"{prefix}_training.png"
    samples_plot_path = f"{prefix}_samples.png"

    device_net, device_wph = parse_devices(args)
    print(f"U-Net on {device_net}, WPH on {device_wph}", flush=True)

    data_torch, spectrum_map = load_data(data_path, M, device_wph, device_net)
    torch.save(spectrum_map.detach().cpu(), spectrum_path)
    print(f"Saved input spectrum to {spectrum_path}", flush=True)

    print("Building WPH operator...", flush=True)
    t0 = time.time()
    wph_op = pw.WPHOp(M, M, J, L=L, dn=dn, device=device_wph)
    wph_op.load_model(p_list=[], dl=dl, dj=dj)
    print(f"Done! ({time.time() - t0:.1f}s)", flush=True)

    print("Computing target WPH coefficients...", flush=True)
    t0 = time.time()
    # Set pywph's auto-normalisation from a single slice. Its cache keeps the
    # batch dimension, so it must not be set from the whole stack.
    _ = wph_op(data_torch[0], norm=norm)
    coeffs = wph_op.apply(data_torch, norm=norm).mean(axis=0)
    print(f"Done! ({time.time() - t0:.1f}s)", flush=True)

    netG = UNet(1, 1).to(device_net)
    netG.apply(weights_init)

    num_iters = 500
    optimizer = optim.Adam(netG.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    losses = train(
        netG,
        wph_op,
        coeffs,
        spectrum_map,
        image_size,
        norm,
        num_iters=num_iters,
        batch_size=200,
        n_chunk=1,
        optimizer=optimizer,
        scheduler=scheduler,
        device_net=device_net,
        device_wph=device_wph,
    )

    torch.save(netG, save_path)
    print(f"Saved trained generator to {save_path}", flush=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.semilogy(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.subplot(122)
    plt.loglog(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    print(f"Saved loss curve to {loss_plot_path}", flush=True)

    make_sample_comparison(
        netG, data_torch, spectrum_map, image_size, samples_plot_path
    )
    print(f"Saved sample comparison to {samples_plot_path}", flush=True)


if __name__ == "__main__":
    main()
