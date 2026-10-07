from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from build_spectrogram_dataset import (
    estimate_num_samples,
    estimate_output_gb,
    make_full_stft_patch_sequences,
    make_sequences,
    normalize_spectrograms,
)


AXES = ("X", "Y", "Z")
NAME_PATTERN = re.compile(r"^(?P<base>.+)_Movement_(?P<axis>[XYZ])_(?P<trial>train\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AccelScansComponents_npy X/Y/Z component files as 3-channel DFGFlow data."
    )
    p.add_argument("--npy-dir", type=str, default="data/AccelScansComponents_npy")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--series-length", type=int, choices=[128, 256, 512, 1024, 2048], default=256)
    p.add_argument("--window-length", type=int, default=256)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--layout", choices=["window_stft", "full_stft_patches"], default="full_stft_patches")
    p.add_argument("--freq-bins", type=int, default=64, help="Only used by --layout full_stft_patches.")
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--representation", choices=["complex", "magnitude"], default="complex")
    p.add_argument("--spec-normalization", choices=["channel", "channel_freq"], default="channel_freq")
    p.add_argument("--max-samples", type=int, default=None, help="Total maximum samples across all complete X/Y/Z trials.")
    p.add_argument("--max-samples-per-trial", type=int, default=None)
    p.add_argument("--max-output-gb", type=float, default=16.0)
    return p.parse_args()


def grouped_axis_files(root: Path) -> list[tuple[tuple[str, str], dict[str, Path]]]:
    groups: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root.glob("*.npy")):
        match = NAME_PATTERN.match(path.name)
        if match is None:
            continue
        key = (match.group("base"), match.group("trial"))
        groups.setdefault(key, {})[match.group("axis")] = path
    complete = [(key, axes) for key, axes in groups.items() if all(axis in axes for axis in AXES)]
    return sorted(complete, key=lambda item: (item[0][0], item[0][1]))


def load_xyz(axes: dict[str, Path]) -> np.ndarray:
    arrays = [np.asarray(np.load(axes[axis]), dtype=np.float32).reshape(-1) for axis in AXES]
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"X/Y/Z lengths differ: {[array.shape for array in arrays]}")
    return np.stack(arrays, axis=1)


def global_xyz_stats(groups: list[tuple[tuple[str, str], dict[str, Path]]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute one dataset-wide scale while retaining trial-to-trial amplitude differences."""
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for _, axes in groups:
        data = load_xyz(axes).astype(np.float64, copy=False)
        count += data.shape[0]
        total += data.sum(axis=0)
        total_sq += np.square(data).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot compute normalization statistics from empty trials.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.npy_dir)
    if not root.exists():
        raise FileNotFoundError(f"npy dir not found: {root}")
    if args.series_length % args.window_length != 0:
        raise ValueError("--series-length must be divisible by --window-length.")
    seq_len = args.series_length // args.window_length
    if args.layout == "full_stft_patches":
        args.n_fft = args.n_fft or 2 * (args.freq_bins - 1)
        args.hop_length = args.hop_length or 1
        expected_freq_bins = args.n_fft // 2 + 1
        if expected_freq_bins < args.freq_bins:
            raise ValueError(f"--n-fft={args.n_fft} only gives {expected_freq_bins} rFFT bins, less than --freq-bins={args.freq_bins}.")
    else:
        args.n_fft = args.n_fft or 32
        args.hop_length = args.hop_length or 8

    groups = grouped_axis_files(root)
    if not groups:
        raise ValueError(f"No complete X/Y/Z groups found in {root}")

    raw_mean, raw_std = global_xyz_stats(groups)

    all_sequences = []
    all_raw = []
    material_ids = []
    trial_ids = []
    remaining = args.max_samples
    estimated_samples = 0
    for (material, trial), axes in groups:
        length = np.load(axes["X"], mmap_mode="r").shape[0]
        per_trial_limit = args.max_samples_per_trial
        if remaining is not None:
            per_trial_limit = remaining if per_trial_limit is None else min(per_trial_limit, remaining)
        count = estimate_num_samples(length, args.series_length, args.stride, per_trial_limit)
        estimated_samples += count
        if remaining is not None:
            remaining -= count
            if remaining <= 0:
                break

    spec_channels = 6 if args.representation == "complex" else 3
    freq_bins = args.freq_bins if args.layout == "full_stft_patches" else args.n_fft // 2 + 1
    time_bins = args.window_length if args.layout == "full_stft_patches" else 1 + max(0, (args.window_length - args.n_fft) // args.hop_length)
    estimated_gb = estimate_output_gb(estimated_samples, seq_len, spec_channels, freq_bins, time_bins)
    if args.max_output_gb > 0 and estimated_gb > args.max_output_gb:
        raise ValueError(
            f"Estimated spectrogram array is {estimated_gb:.1f} GB "
            f"({estimated_samples} samples, shape [N,{seq_len},{spec_channels},{freq_bins},{time_bins}]). "
            "Increase --stride, set --max-samples, or pass --max-output-gb 0 to disable this guard."
        )
    print(
        f"building {estimated_samples} samples from {len(groups)} complete trials with shape "
        f"[N,{seq_len},{spec_channels},{freq_bins},{time_bins}] (estimated {estimated_gb:.2f} GB float32)"
    )

    remaining = args.max_samples
    for (material, trial), axes in groups:
        per_trial_limit = args.max_samples_per_trial
        if remaining is not None:
            per_trial_limit = remaining if per_trial_limit is None else min(per_trial_limit, remaining)
            if per_trial_limit <= 0:
                break
        data = ((load_xyz(axes) - raw_mean[None, :]) / raw_std[None, :]).astype(np.float32)
        if args.layout == "full_stft_patches":
            sequences, raw_sequences = make_full_stft_patch_sequences(
                data,
                series_length=args.series_length,
                seq_len=seq_len,
                patch_time_bins=args.window_length,
                stride=args.stride,
                n_fft=args.n_fft,
                freq_bins=args.freq_bins,
                max_samples=per_trial_limit,
                representation=args.representation,
            )
        else:
            sequences, raw_sequences = make_sequences(
                data,
                seq_len=seq_len,
                window_length=args.window_length,
                stride=args.stride,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                max_samples=per_trial_limit,
                representation=args.representation,
            )
        all_sequences.append(sequences)
        all_raw.append(raw_sequences)
        material_ids.append(np.full(sequences.shape[0], material, dtype=object))
        trial_ids.append(np.full(sequences.shape[0], trial, dtype=object))
        if remaining is not None:
            remaining -= sequences.shape[0]
        print(f"{material}/{trial}: {sequences.shape[0]} samples")

    if not all_sequences:
        raise ValueError("No samples were created.")
    sequences = np.concatenate(all_sequences, axis=0)
    raw_sequences = np.concatenate(all_raw, axis=0)
    sample_material_ids = np.concatenate(material_ids, axis=0)
    sample_trial_ids = np.concatenate(trial_ids, axis=0)
    sample_record_ids = np.asarray(
        [f"{material}/{trial}" for material, trial in zip(sample_material_ids, sample_trial_ids)],
        dtype=object,
    )
    sequences, spec_mean, spec_std = normalize_spectrograms(sequences, args.spec_normalization)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(f".{out.stem}.tmp.npz")
    if tmp_out.exists():
        tmp_out.unlink()
    np.savez_compressed(
        tmp_out,
        spectrograms=sequences,
        raw_sequences=raw_sequences,
        raw_mean=raw_mean,
        raw_std=raw_std,
        sample_material_ids=sample_material_ids,
        sample_trial_ids=sample_trial_ids,
        sample_record_ids=sample_record_ids,
        spec_mean=spec_mean,
        spec_std=spec_std,
        spec_normalization=np.array(args.spec_normalization),
        normalization=np.array("dataset_channel_standard"),
        columns=np.array(["x", "y", "z"], dtype=object),
        representation=np.array(args.representation),
        original_channels=np.array(3),
        spectrogram_layout=np.array(args.layout),
        series_length=np.array(args.series_length),
        window_length=np.array(args.window_length),
        patch_time_bins=np.array(args.window_length),
        freq_bins=np.array(args.freq_bins if args.layout == "full_stft_patches" else sequences.shape[3]),
        seq_len=np.array(seq_len),
        stride=np.array(args.stride),
        n_fft=np.array(args.n_fft),
        hop_length=np.array(args.hop_length),
    )
    tmp_out.replace(out)
    print(f"saved {sequences.shape} to {out}")
    print(f"raw_sequences={raw_sequences.shape}, columns=x,y,z")


if __name__ == "__main__":
    main()
