from __future__ import annotations

import torch
from torch.nn import functional as F


def kl_normal(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).square()) / var_p.clamp_min(1e-8) - 1.0)
    return kl.mean()


def smooth_direct_target(target: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    b, n, c, f, w = target.shape
    flat = target.reshape(b * n, c, f, w)
    smooth = F.avg_pool2d(flat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return smooth.view_as(target)


def total_variation_loss(specs: torch.Tensor) -> torch.Tensor:
    freq_tv = (specs[:, :, :, 1:, :] - specs[:, :, :, :-1, :]).abs().mean()
    time_tv = (specs[:, :, :, :, 1:] - specs[:, :, :, :, :-1]).abs().mean()
    seq_tv = (specs[:, 1:] - specs[:, :-1]).abs().mean() if specs.shape[1] > 1 else specs.new_tensor(0.0)
    return freq_tv + time_tv + 0.25 * seq_tv


def gradient_detail_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_freq = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    target_freq = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    pred_time = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    target_time = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    pred_seq = pred[:, 1:] - pred[:, :-1] if pred.shape[1] > 1 else None
    target_seq = target[:, 1:] - target[:, :-1] if target.shape[1] > 1 else None
    loss = F.l1_loss(pred_freq, target_freq) + F.l1_loss(pred_time, target_time)
    if pred_seq is not None and target_seq is not None:
        loss = loss + 0.25 * F.l1_loss(pred_seq, target_seq)
    return loss



def frequency_weighted_l1(pred: torch.Tensor, target: torch.Tensor, high_boost: float = 2.0) -> torch.Tensor:
    freq_bins = pred.shape[-2]
    weights = torch.linspace(1.0, high_boost, freq_bins, device=pred.device, dtype=pred.dtype)
    weights = weights.view(1, 1, 1, freq_bins, 1)
    return (weights * (pred - target).abs()).mean()

def denormalize_spectrogram(
    specs: torch.Tensor,
    spec_mean: torch.Tensor | None,
    spec_std: torch.Tensor | None,
) -> torch.Tensor:
    if spec_mean is None or spec_std is None:
        return specs
    mean = spec_mean.to(specs.device, specs.dtype)
    std = spec_std.to(specs.device, specs.dtype)
    if mean.ndim == 1:
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    elif mean.ndim == 2:
        mean = mean.view(1, 1, mean.shape[0], mean.shape[1], 1)
        std = std.view(1, 1, std.shape[0], std.shape[1], 1)
    elif mean.ndim == 3:
        mean = mean.view(1, 1, *mean.shape)
        std = std.view(1, 1, *std.shape)
    else:
        raise ValueError(f"Unsupported spec_mean shape: {tuple(mean.shape)}")
    return specs * std + mean


def normalize_spectrogram(
    specs: torch.Tensor,
    spec_mean: torch.Tensor | None,
    spec_std: torch.Tensor | None,
) -> torch.Tensor:
    if spec_mean is None or spec_std is None:
        return specs
    mean = spec_mean.to(specs.device, specs.dtype)
    std = spec_std.to(specs.device, specs.dtype)
    if mean.ndim == 1:
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    elif mean.ndim == 2:
        mean = mean.view(1, 1, mean.shape[0], mean.shape[1], 1)
        std = std.view(1, 1, std.shape[0], std.shape[1], 1)
    elif mean.ndim == 3:
        mean = mean.view(1, 1, *mean.shape)
        std = std.view(1, 1, *std.shape)
    else:
        raise ValueError(f"Unsupported spec_mean shape: {tuple(mean.shape)}")
    return (specs - mean) / std.clamp_min(1e-6)


def inverse_complex_spectrogram_sequence(
    specs: torch.Tensor,
    *,
    original_channels: int,
    window_length: int,
    n_fft: int,
    hop_length: int,
    spec_mean: torch.Tensor | None = None,
    spec_std: torch.Tensor | None = None,
) -> torch.Tensor:
    specs = denormalize_spectrogram(specs, spec_mean, spec_std)
    batch, seq_len, _, _, frame_count = specs.shape
    window = torch.hann_window(n_fft, periodic=False, device=specs.device, dtype=specs.dtype)
    pieces = []
    for step_idx in range(seq_len):
        channel_pieces = []
        for channel_idx in range(original_channels):
            real = specs[:, step_idx, 2 * channel_idx]
            imag = specs[:, step_idx, 2 * channel_idx + 1]
            complex_spec = torch.complex(real, imag)
            tile = torch.zeros(batch, window_length, device=specs.device, dtype=specs.dtype)
            norm = torch.zeros_like(tile)
            for frame_idx in range(frame_count):
                frame = torch.fft.irfft(complex_spec[:, :, frame_idx], n=n_fft)
                start = frame_idx * hop_length
                end = min(start + n_fft, window_length)
                valid = end - start
                if valid > 0:
                    tile[:, start:end] = tile[:, start:end] + frame[:, :valid] * window[:valid]
                    norm[:, start:end] = norm[:, start:end] + window[:valid].square()
            channel_pieces.append(tile / norm.clamp_min(1e-6))
        pieces.append(torch.stack(channel_pieces, dim=1))
    return torch.stack(pieces, dim=1)


def inverse_centered_complex_patch_sequence(
    specs: torch.Tensor,
    *,
    original_channels: int,
    n_fft: int,
    spec_mean: torch.Tensor | None = None,
    spec_std: torch.Tensor | None = None,
) -> torch.Tensor:
    specs = denormalize_spectrogram(specs, spec_mean, spec_std)
    batch, seq_len, _, freq_count, patch_time = specs.shape
    full_length = seq_len * patch_time
    expected_freq = n_fft // 2 + 1
    if freq_count < expected_freq:
        pad = specs.new_zeros(batch, seq_len, specs.shape[2], expected_freq - freq_count, patch_time)
        specs = torch.cat([specs, pad], dim=3)
    elif freq_count > expected_freq:
        specs = specs[:, :, :, :expected_freq]
    real = specs[:, :, 0::2].permute(0, 2, 3, 1, 4).reshape(batch * original_channels, expected_freq, full_length)
    imag = specs[:, :, 1::2].permute(0, 2, 3, 1, 4).reshape(batch * original_channels, expected_freq, full_length)
    frames = torch.fft.irfft(torch.complex(real, imag), n=n_fft, dim=1)
    window = torch.hann_window(n_fft, periodic=False, device=specs.device, dtype=specs.dtype)
    frames = frames * window[None, :, None]
    padded_length = full_length + n_fft - 1
    folded = F.fold(frames, output_size=(1, padded_length), kernel_size=(1, n_fft), stride=(1, 1)).squeeze(2).squeeze(1)
    norm_frames = window.square()[None, :, None].expand(1, n_fft, full_length)
    norm = F.fold(norm_frames, output_size=(1, padded_length), kernel_size=(1, n_fft), stride=(1, 1)).squeeze(2).squeeze(1)
    half = n_fft // 2
    raw = folded[:, half : half + full_length] / norm[:, half : half + full_length].clamp_min(1e-6)
    return raw.view(batch, original_channels, seq_len, patch_time).permute(0, 2, 1, 3)


def centered_complex_patch_sequence(
    raw: torch.Tensor,
    *,
    n_fft: int,
    freq_bins: int,
    spec_mean: torch.Tensor | None = None,
    spec_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable counterpart of the centered-STFT preprocessing path."""
    batch, seq_len, channels, patch_time = raw.shape
    full_length = seq_len * patch_time
    signal = raw.permute(0, 2, 1, 3).reshape(batch, channels, full_length).float()
    half = n_fft // 2
    padded = F.pad(signal, (half, n_fft - half - 1), mode="reflect")
    frames = padded.unfold(-1, n_fft, 1)
    window = torch.hann_window(n_fft, periodic=False, device=raw.device, dtype=signal.dtype)
    spectrum = torch.fft.rfft(frames * window, n=n_fft, dim=-1)
    if spectrum.shape[-1] < freq_bins:
        spectrum = F.pad(spectrum, (0, freq_bins - spectrum.shape[-1]))
    spectrum = spectrum[..., :freq_bins]
    real_imag = torch.stack((spectrum.real, spectrum.imag), dim=2)
    specs = real_imag.permute(0, 1, 2, 4, 3).reshape(batch, channels * 2, freq_bins, full_length)
    specs = specs.view(batch, channels * 2, freq_bins, seq_len, patch_time).permute(0, 3, 1, 2, 4)
    specs = normalize_spectrogram(specs, spec_mean, spec_std)
    return specs.to(raw.dtype)
