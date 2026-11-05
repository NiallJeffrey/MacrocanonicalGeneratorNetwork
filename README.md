# Macrocanonical Generator Networks

Code for the paper

> **Macrocanonical Generator Networks: Data-Efficient Neural Surrogates for
> Amortized Physics Simulation**<br>
> Niall Jeffrey and Benjamin D. Wandelt, NeurIPS 2026<br>
> [arXiv:2611.05157](https://arxiv.org/abs/2611.05157)

![Macrocanonical generator networks overview](summary_plot.png)

A macrocanonical generator maps a Gaussian random field carrying the target's
power spectrum to a new realisation whose wavelet phase harmonic (WPH)
statistics match, in expectation, those of a small pool of simulator outputs,
in some cases a single one. Once trained, each new sample takes one forward
pass.

Two benchmarks are included, both at 128 x 128 with periodic boundaries:

| Benchmark | Target | Training fields |
| --- | --- | --- |
| `mhd_turbulence` | Magnetohydrodynamic turbulence simulation | 1 |
| `cosmic_web` | Quijote N-body density slices | 100 |

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.8 to 3.11. `pywph` is pinned to 1.0.1, whose
`load_model(p_list=...)` interface the training scripts use; it predates
NumPy 1.24, so NumPy is pinned below that.

## Training

```bash
python train_mhd_turbulence.py
python train_cosmic_web.py
```

`train_mhd_turbulence.py` reads `data/mhd_sim_demo.npy`, which is included.
`train_cosmic_web.py` expects `data/Quijote_fidu_300.npy`, which is not; see
[Data](#data).

Each run writes `<prefix>_netG` (the generator), `<prefix>_spectrum.pt` (the
input power spectrum), a loss curve and a sample figure, where `<prefix>` is
`mhd_turbulence_<timestamp>` or `cosmic_web_<timestamp>`. Pass `--seed` for
a reproducible run.

Training needs CUDA. The generator and the WPH operator are placed on two
GPUs, `--gpus NET WPH` (default `0 1`); passing the same index twice puts
both on one GPU, which may not have enough memory at batch size 200. The
training budget is set by `num_iters` in each script's `main()`: 6000
iterations for `mhd_turbulence` and 500 for `cosmic_web`, at batch size 200.

## Sampling

```bash
python generate_mhd_turbulence.py --model mhd_turbulence_<timestamp>_netG --n-samples 8
python generate_cosmic_web.py --model cosmic_web_<timestamp>_netG --n-samples 8
```

The matching `<prefix>_spectrum.pt` is found next to the model, or can be
given with `--spectrum`. Writes a PNG grid, and a `.npy` array if
`--samples-output` is given. Sampling runs on a GPU if one is available,
otherwise on the CPU.

## Data

`data/mhd_sim_demo.npy` is a 512 x 512 map of simulated dust emission from
the magnetohydrodynamic simulation of the diffuse interstellar medium used by
Regaldo-Saint Blancard, Levrier, Allys, Bellomi and Boulanger (2020),
*Statistical description of dust polarized emission from the diffuse
interstellar medium: A RWST approach*, A&A 642, A217,
[arXiv:2007.08242](https://arxiv.org/abs/2007.08242),
[doi:10.1051/0004-6361/202038044](https://doi.org/10.1051/0004-6361/202038044).
The map is stored as a complex-valued array; `train_mhd_turbulence.py` takes
its modulus, applies a `log10` transform, mean-pools to 128 x 128, removes a
smooth large-scale component and normalises. Please cite that paper if you
use this file.

The cosmic-web benchmark uses slices from the Quijote simulations
(Villaescusa-Navarro et al. 2020, ApJS 250, 2), which must be obtained from
https://quijote-simulations.readthedocs.io under their terms of use.
`data/Quijote_fidu_300.npy` should be an array of shape `(N, 256, 256)`
holding 2-D density-contrast slices (`delta = rho / rho_mean - 1`) from the
fiducial suite, of which the first 100 are used. The script applies
`log10(delta + 1)`, mean-pools to 128 x 128 and normalises; see Appendix C of
the paper.

## Citation

```bibtex
@misc{jeffrey2026macrocanonical,
  title         = {Macrocanonical Generator Networks: Data-Efficient Neural
                   Surrogates for Amortized Physics Simulation},
  author        = {Jeffrey, Niall and Wandelt, Benjamin D.},
  year          = {2026},
  eprint        = {2611.05157},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Licence

MIT, see [LICENSE](LICENSE).
