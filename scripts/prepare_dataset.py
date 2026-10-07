from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "ETTh1": "ETTh1.csv",
    "energy": "energy_data.csv",
    "stock": "stock_data.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--series-length", type=int, choices=[128, 256, 512, 1024, 2048], required=True)
    p.add_argument("--window-length", type=int, default=64, help="Raw samples per STFT patch. Set equal to --series-length for one spectrogram per sample.")
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--layout", choices=["window_stft", "full_stft_patches"], default="window_stft")
    p.add_argument("--freq-bins", type=int, default=64, help="Only used by --layout full_stft_patches.")
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--channels", type=str, default="all")
    p.add_argument("--representation", choices=["complex", "magnitude"], default="complex")
    p.add_argument("--spec-normalization", choices=["channel", "channel_freq"], default="channel")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-output-gb", type=float, default=16.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    csv_path = data_dir / DATASETS[args.dataset]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.layout == "full_stft_patches":
        stem = (
            f"{args.dataset}_{args.representation}_patchstft"
            f"_len{args.series_length}_patch{args.window_length}_freq{args.freq_bins}_stride{args.stride}"
        )
    else:
        stem = f"{args.dataset}_{args.representation}_len{args.series_length}_win{args.window_length}_stride{args.stride}"
    if args.channels != "all":
        safe_channels = args.channels.replace(",", "-").replace(" ", "")
        stem += f"_{safe_channels}"
    out_path = out_dir / f"{stem}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"exists: {out_path}")
        print("use --overwrite to rebuild")
        return

    cmd = [
        sys.executable,
        "scripts/build_spectrogram_dataset.py",
        "--csv",
        str(csv_path),
        "--out",
        str(out_path),
        "--series-length",
        str(args.series_length),
        "--window-length",
        str(args.window_length),
        "--stride",
        str(args.stride),
        "--layout",
        args.layout,
    ]
    if args.layout == "full_stft_patches":
        cmd += ["--freq-bins", str(args.freq_bins)]
    if args.n_fft is not None:
        cmd += [
        "--n-fft",
        str(args.n_fft),
        ]
    if args.hop_length is not None:
        cmd += [
        "--hop-length",
        str(args.hop_length),
        ]
    cmd += [
        "--channels",
        args.channels,
        "--representation",
        args.representation,
        "--spec-normalization",
        args.spec_normalization,
        "--max-output-gb",
        str(args.max_output_gb),
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
