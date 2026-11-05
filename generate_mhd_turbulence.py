#!/usr/bin/env python
"""Generate MHD turbulence samples from a generator trained with
train_mhd_turbulence.py, and save them as a PNG grid.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Imported into __main__ so torch.load can unpickle checkpoints saved by the
# training script.
from train_mhd_turbulence import (  # noqa: F401
    CircularUpsample2d,
    DoubleConv,
    Down,
    DustGenerator,
    OutConv,
    UNet,
    UNet2,
    Up,
    gaussian_mock_batch,
)


IMAGE_SIZE = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="Path to the <prefix>_netG checkpoint written by training.",
    )
    p.add_argument(
        "--spectrum",
        default=None,
        help=(
            "Path to a saved spectrum tensor (.pt) produced by training. "
            "If omitted, a sibling file ``<prefix>_spectrum.pt`` next to the "
            "model is used."
        ),
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of samples to generate.",
    )
    p.add_argument(
        "--n-cols",
        type=int,
        default=4,
        help="Columns in the output figure grid.",
    )
    p.add_argument(
        "--output",
        default="mhd_turbulence_generated.png",
        help="Output PNG path for the grid figure.",
    )
    p.add_argument(
        "--samples-output",
        default=None,
        help="Optional .npy path to also save the raw generated tensor of shape (N, 128, 128).",
    )
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for generation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    p.add_argument("--vmin", type=float, default=-0.4)
    p.add_argument("--vmax", type=float, default=0.4)
    return p.parse_args()


def load_spectrum(args, device):
    """Load ``--spectrum``, or else the ``<prefix>_spectrum.pt`` next to ``--model``."""
    if args.spectrum is not None:
        path, source = args.spectrum, "--spectrum"
    else:
        sibling = _sibling_spectrum_path(args.model)
        if sibling is None:
            raise SystemExit(
                "Could not auto-derive a spectrum path: --model name does not "
                "end in '_netG'. Pass --spectrum explicitly."
            )
        path, source = sibling, "sibling of --model"

    if not os.path.exists(path):
        raise SystemExit(
            f"Spectrum file not found at {path} ({source}). "
            f"Pass --spectrum to point at one explicitly."
        )

    print(f"Loading spectrum from {path} ({source})", flush=True)
    return torch.load(path, map_location=device, weights_only=True)


def _sibling_spectrum_path(model_path):
    if model_path.endswith("_netG"):
        return model_path[: -len("_netG")] + "_spectrum.pt"
    return None


def generate_samples(netG, spectrum_map, n_samples, image_size):
    """Return ``n_samples`` independent samples as an array of shape
    ``(n_samples, image_size, image_size)``."""
    device = next(netG.parameters()).device
    netG.eval()
    with torch.no_grad():
        noise = gaussian_mock_batch(n_samples, spectrum_map, image_size, device)
        noise_in = noise.reshape(n_samples, 1, image_size, image_size)
        # DustGenerator applies its own residual, F(x) + G(x) * x.
        samples = netG(noise_in).reshape(n_samples, image_size, image_size)
    return samples.cpu().numpy()


def save_sample_grid(samples, save_path, n_cols, vmin, vmax):
    """Save the samples as a PNG grid."""
    n_samples = samples.shape[0]
    n_rows = (n_samples + n_cols - 1) // n_cols

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "mathtext.fontset": "cm",
        "savefig.dpi": 300,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 1.5 + 0.6, n_rows * 1.5 + 0.3),
            constrained_layout=True,
            squeeze=False,
        )
        for i in range(n_rows * n_cols):
            r, c = divmod(i, n_cols)
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i < n_samples:
                im = ax.imshow(
                    samples[i],
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
            else:
                ax.set_visible(False)

        cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.015)
        cbar.set_label(r"normalised field")
        cbar.outline.set_visible(False)

        fig.suptitle("Generated MHD turbulence fields", fontsize=10, y=1.02)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Generating {args.n_samples} MHD turbulence samples on {device}", flush=True)

    spectrum_map = load_spectrum(args, device)

    print(f"Loading trained generator from {args.model}", flush=True)
    netG = torch.load(args.model, map_location=device, weights_only=False)
    if not isinstance(netG, DustGenerator):
        # Fall back: checkpoint was a state_dict, not a whole module.
        state = netG
        netG = DustGenerator().to(device)
        netG.load_state_dict(state)

    samples = generate_samples(netG, spectrum_map, args.n_samples, IMAGE_SIZE)
    print(
        f"Generated samples: shape={samples.shape}, "
        f"range=({samples.min():.3f}, {samples.max():.3f}), "
        f"std={samples.std():.3f}",
        flush=True,
    )

    save_sample_grid(samples, args.output, args.n_cols, args.vmin, args.vmax)
    print(f"Saved MHD turbulence sample grid to {args.output}", flush=True)

    if args.samples_output is not None:
        np.save(args.samples_output, samples)
        print(f"Saved raw samples tensor to {args.samples_output}", flush=True)


if __name__ == "__main__":
    main()
