from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--series-length", type=int, default=None, help="Total raw time-series length per sample, e.g. 256/512/1024/2048")
    p.add_argument("--seq-len", type=int, default=None, help="Number of STFT patches per sample. Inferred from --series-length when omitted.")
    p.add_argument("--window-length", "--tile-length", dest="window_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument(
        "--layout",
        choices=["window_stft", "full_stft_patches"],
        default="window_stft",
        help=(
            "window_stft: STFT each raw window independently; "
            "full_stft_patches: experimental stretched layout that makes one full-series map then splits it."
        ),
    )
    p.add_argument("--freq-bins", type=int, default=64, help="Frequency bins per patch for --layout full_stft_patches.")
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-output-gb", type=float, default=16.0, help="Stop before building arrays larger than this estimate. Use 0 to disable.")
    p.add_argument("--channels", type=str, default="all", help="'all' or comma-separated numeric column names")
    p.add_argument("--representation", choices=["complex", "magnitude"], default="complex")
    p.add_argument(
        "--spec-normalization",
        choices=["channel", "channel_freq"],
        default="channel",
        help="channel is the default per-channel scale. channel_freq is experimental and can amplify weak high-frequency bins.",
    )
    return p.parse_args()


def load_numeric_csv(path: str | Path, channel_spec: str) -> tuple[np.ndarray, list[str]]:
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = list(raw.dtype.names or [])
    numeric_names = []
    columns = []
    requested = None if channel_spec == "all" else {name.strip() for name in channel_spec.split(",") if name.strip()}
    for name in names:
        values = raw[name]
        if requested is not None and name not in requested:
            continue
        try:
            numeric = values.astype(np.float32)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric).all():
            numeric_names.append(name)
            columns.append(numeric)
    if not columns:
        raise ValueError(f"No numeric columns found in {path}")
    data = np.stack(columns, axis=1)
    return data, numeric_names


def normalization_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel statistics that can restore the original units."""
    mean = data.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = data.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def normalize(data: np.ndarray) -> np.ndarray:
    mean, std = normalization_stats(data)
    return ((data - mean[None, :]) / std[None, :]).astype(np.float32)


def stft_complex(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if signal.shape[0] < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.shape[0]))
    frame_count = 1 + max(0, (signal.shape[0] - n_fft) // hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for frame_idx in range(frame_count):
        start = frame_idx * hop_length
        frame = signal[start : start + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        spectrum = np.fft.rfft(frame * window)
        frames.append(spectrum)
    spec = np.stack(frames, axis=1)
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)


def centered_stft_complex(signal: np.ndarray, n_fft: int, freq_bins: int) -> np.ndarray:
    """Returns a centered STFT-like map [2,F,L] with one frame per raw sample.

    This is used for the full-series patch layout:
      raw length 1024 -> full map [F,1024] -> 16 patches of [F,64].
    """
    half = n_fft // 2
    padded = np.pad(signal, (half, n_fft - half - 1), mode="reflect")
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for center_idx in range(signal.shape[0]):
        frame = padded[center_idx : center_idx + n_fft]
        spectrum = np.fft.rfft(frame * window)
        if spectrum.shape[0] < freq_bins:
            spectrum = np.pad(spectrum, (0, freq_bins - spectrum.shape[0]))
        frames.append(spectrum[:freq_bins])
    spec = np.stack(frames, axis=1)
    return np.stack([spec.real, spec.imag], axis=0).astype(np.float32)


def full_centered_spectrogram_maps(
    data: np.ndarray,
    n_fft: int,
    freq_bins: int,
    representation: str,
) -> np.ndarray:
    """Computes full-length spectrogram maps once for the whole CSV.

    Output shape is [spec_channels,F,total_time]. For complex representation,
    spec_channels = original_channels * 2.
    """
    channel_maps = []
    for channel_idx in range(data.shape[1]):
        complex_spec = centered_stft_complex(data[:, channel_idx], n_fft=n_fft, freq_bins=freq_bins)
        if representation == "complex":
            channel_maps.extend([complex_spec[0], complex_spec[1]])
        else:
            channel_maps.append(np.sqrt(complex_spec[0] ** 2 + complex_spec[1] ** 2))
    return np.stack(channel_maps, axis=0).astype(np.float32)


def stft_magnitude(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    complex_pair = stft_complex(signal, n_fft=n_fft, hop_length=hop_length)
    spec = np.sqrt(complex_pair[0] ** 2 + complex_pair[1] ** 2)
    return spec


def normalize_spectrograms(sequences: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "channel_freq":
        mean = sequences.mean(axis=(0, 1, 4), keepdims=True)
        std = sequences.std(axis=(0, 1, 4), keepdims=True)
        stat_shape = (sequences.shape[2], sequences.shape[3])
    else:
        mean = sequences.mean(axis=(0, 1, 3, 4), keepdims=True)
        std = sequences.std(axis=(0, 1, 3, 4), keepdims=True)
        stat_shape = (sequences.shape[2],)
    std = np.maximum(std, 1e-6)
    normalized = (sequences - mean) / std
    return normalized.astype(np.float32), mean.reshape(stat_shape).astype(np.float32), std.reshape(stat_shape).astype(np.float32)


def make_sequences(
    data: np.ndarray,
    seq_len: int,
    window_length: int,
    stride: int,
    n_fft: int,
    hop_length: int,
    max_samples: int | None,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    total_length = seq_len * window_length
    spec_samples = []
    raw_samples = []
    for start in range(0, data.shape[0] - total_length + 1, stride):
        tiles = []
        raw_tiles = []
        for step in range(seq_len):
            tile_start = start + step * window_length
            tile_data = data[tile_start : tile_start + window_length]
            raw_tiles.append(tile_data.T)
            channel_specs = []
            for channel_idx in range(tile_data.shape[1]):
                if representation == "complex":
                    complex_spec = stft_complex(tile_data[:, channel_idx], n_fft=n_fft, hop_length=hop_length)
                    channel_specs.extend([complex_spec[0], complex_spec[1]])
                else:
                    channel_specs.append(stft_magnitude(tile_data[:, channel_idx], n_fft=n_fft, hop_length=hop_length))
            tiles.append(np.stack(channel_specs, axis=0))
        spec_samples.append(np.stack(tiles, axis=0))
        raw_samples.append(np.stack(raw_tiles, axis=0))
        if max_samples is not None and len(spec_samples) >= max_samples:
            break
    if not spec_samples:
        raise ValueError("No samples were created. Try smaller --seq-len or --tile-length.")
    return np.stack(spec_samples, axis=0).astype(np.float32), np.stack(raw_samples, axis=0).astype(np.float32)


def make_full_stft_patch_sequences(
    data: np.ndarray,
    series_length: int,
    seq_len: int,
    patch_time_bins: int,
    stride: int,
    n_fft: int,
    freq_bins: int,
    max_samples: int | None,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    if series_length != seq_len * patch_time_bins:
        raise ValueError("--series-length must equal --seq-len * --window-length in full_stft_patches mode.")
    full_maps = full_centered_spectrogram_maps(
        data,
        n_fft=n_fft,
        freq_bins=freq_bins,
        representation=representation,
    )
    spec_samples = []
    raw_samples = []
    for start in range(0, data.shape[0] - series_length + 1, stride):
        series = data[start : start + series_length]
        full_map = full_maps[:, :, start : start + series_length]
        tiles = []
        raw_tiles = []
        for step in range(seq_len):
            patch_start = step * patch_time_bins
            patch_end = patch_start + patch_time_bins
            tiles.append(full_map[:, :, patch_start:patch_end])
            raw_tiles.append(series[patch_start:patch_end].T)
        spec_samples.append(np.stack(tiles, axis=0))
        raw_samples.append(np.stack(raw_tiles, axis=0))
        if max_samples is not None and len(spec_samples) >= max_samples:
            break
    if not spec_samples:
        raise ValueError("No samples were created. Try smaller --series-length or --stride.")
    return np.stack(spec_samples, axis=0).astype(np.float32), np.stack(raw_samples, axis=0).astype(np.float32)


def estimate_num_samples(data_length: int, series_length: int, stride: int, max_samples: int | None) -> int:
    if data_length < series_length:
        return 0
    total = 1 + (data_length - series_length) // stride
    return min(total, max_samples) if max_samples is not None else total


def estimate_output_gb(num_samples: int, seq_len: int, channels: int, freq_bins: int, time_bins: int) -> float:
    elements = num_samples * seq_len * channels * freq_bins * time_bins
    return elements * np.dtype(np.float32).itemsize / (1024**3)


def main() -> None:
    args = parse_args()
    if args.seq_len is None and args.series_length is None:
        raise ValueError("Provide --series-length or --seq-len.")
    if args.series_length is not None:
        if args.series_length % args.window_length != 0:
            raise ValueError("--series-length must be divisible by --window-length for exact generation length.")
        inferred_seq_len = args.series_length // args.window_length
        if args.seq_len is not None and args.seq_len != inferred_seq_len:
            raise ValueError(
                f"--seq-len={args.seq_len} conflicts with --series-length/--window-length={inferred_seq_len}."
            )
        args.seq_len = inferred_seq_len
    assert args.seq_len is not None
    series_length = args.seq_len * args.window_length
    if args.layout == "full_stft_patches":
        args.n_fft = args.n_fft or 2 * (args.freq_bins - 1)
        args.hop_length = args.hop_length or 1
        expected_freq_bins = args.n_fft // 2 + 1
        if expected_freq_bins < args.freq_bins:
            raise ValueError(f"--n-fft={args.n_fft} only gives {expected_freq_bins} rFFT bins, less than --freq-bins={args.freq_bins}.")
    else:
        args.n_fft = args.n_fft or 32
        args.hop_length = args.hop_length or 8
    data, names = load_numeric_csv(args.csv, args.channels)
    raw_mean, raw_std = normalization_stats(data)
    data = ((data - raw_mean[None, :]) / raw_std[None, :]).astype(np.float32)
    spec_channels = len(names) * 2 if args.representation == "complex" else len(names)
    sample_count = estimate_num_samples(data.shape[0], series_length, args.stride, args.max_samples)
    freq_bins = args.freq_bins if args.layout == "full_stft_patches" else args.n_fft // 2 + 1
    time_bins = args.window_length if args.layout == "full_stft_patches" else 1 + max(0, (args.window_length - args.n_fft) // args.hop_length)
    estimated_gb = estimate_output_gb(sample_count, args.seq_len, spec_channels, freq_bins, time_bins)
    if args.max_output_gb > 0 and estimated_gb > args.max_output_gb:
        raise ValueError(
            f"Estimated spectrogram array is {estimated_gb:.1f} GB "
            f"({sample_count} samples, shape [N,{args.seq_len},{spec_channels},{freq_bins},{time_bins}]). "
            "Increase --stride, set --max-samples, or pass --max-output-gb 0 to disable this guard."
        )
    print(
        f"building {sample_count} samples with shape [N,{args.seq_len},{spec_channels},{freq_bins},{time_bins}] "
        f"(estimated {estimated_gb:.2f} GB float32)"
    )
    if args.layout == "full_stft_patches":
        sequences, raw_sequences = make_full_stft_patch_sequences(
            data,
            series_length=series_length,
            seq_len=args.seq_len,
            patch_time_bins=args.window_length,
            stride=args.stride,
            n_fft=args.n_fft,
            freq_bins=args.freq_bins,
            max_samples=args.max_samples,
            representation=args.representation,
        )
    else:
        sequences, raw_sequences = make_sequences(
            data,
            seq_len=args.seq_len,
            window_length=args.window_length,
            stride=args.stride,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            max_samples=args.max_samples,
            representation=args.representation,
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
        spec_mean=spec_mean,
        spec_std=spec_std,
        spec_normalization=np.array(args.spec_normalization),
        normalization=np.array("dataset_channel_standard"),
        columns=np.array(names, dtype=object),
        representation=np.array(args.representation),
        original_channels=np.array(len(names)),
        spectrogram_layout=np.array(args.layout),
        series_length=np.array(series_length),
        window_length=np.array(args.window_length),
        patch_time_bins=np.array(args.window_length),
        freq_bins=np.array(args.freq_bins if args.layout == "full_stft_patches" else sequences.shape[3]),
        seq_len=np.array(args.seq_len),
        stride=np.array(args.stride),
        n_fft=np.array(args.n_fft),
        hop_length=np.array(args.hop_length),
    )
    tmp_out.replace(out)
    print(f"saved {sequences.shape} to {out}")
    print(
        f"series_length={series_length}, window_length={args.window_length}, "
        f"stride={args.stride}, seq_len={args.seq_len}, layout={args.layout}, "
        f"patch_shape={sequences.shape[3]}x{sequences.shape[4]}"
    )
    print(f"columns: {', '.join(names)}")


if __name__ == "__main__":
    main()
