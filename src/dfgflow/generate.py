"""Generate raw-unit waveforms with optional 1D plots and no evaluation dependencies."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .config import config_from_dict
from .model import DirectDiffModel
from .metadata import load_preprocessing_metadata
from .losses import inverse_centered_complex_patch_sequence, inverse_complex_spectrogram_sequence


def restore_waveform(specs: torch.Tensor, meta: dict) -> torch.Tensor:
    required = ('representation', 'spectrogram_layout', 'original_channels', 'n_fft',
                'spec_mean', 'spec_std', 'raw_mean', 'raw_std')
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f'Missing reconstruction metadata: {missing}. Use --metadata with the training NPZ,.')
    if meta['representation'] != 'complex':
        raise ValueError('Waveform generation requires complex STFT with real/imaginary channels; magnitude-only checkpoints are not supported.')
    def tensor(key):
        return torch.as_tensor(meta[key], dtype=specs.dtype, device=specs.device)
    kwargs = dict(original_channels=int(meta['original_channels']), n_fft=int(meta['n_fft']),
                  spec_mean=tensor('spec_mean'), spec_std=tensor('spec_std'))
    if specs.shape[2] != 2 * kwargs['original_channels']:
        raise ValueError('Spectrogram channel count does not match reconstruction metadata.')
    if meta['spectrogram_layout'] == 'full_stft_patches':
        if int(meta.get('hop_length', 1)) != 1:
            raise ValueError('Full-STFT patch reconstruction requires hop_length=1.')
        wave = inverse_centered_complex_patch_sequence(specs, **kwargs)
    elif meta['spectrogram_layout'] == 'window_stft':
        wave = inverse_complex_spectrogram_sequence(specs, window_length=int(meta['window_length']),
                                                    hop_length=int(meta['hop_length']), **kwargs)
    else:
        raise ValueError(f"Unsupported layout: {meta['spectrogram_layout']}")
    wave = wave * tensor('raw_std').reshape(1, 1, -1, 1) + tensor('raw_mean').reshape(1, 1, -1, 1)
    # [batch, patches, channels, patch_time] -> [batch, time, channels]
    return wave.permute(0, 1, 3, 2).reshape(wave.shape[0], -1, wave.shape[2])


def save_timeseries_plots(series: np.ndarray, directory: Path, names=None) -> None:
    """One PNG per generated sample, with one 1D subplot per channel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(series):
        channels = sample.shape[1]
        fig, axes = plt.subplots(channels, 1, figsize=(12, max(2.5, 1.8 * channels)),
                                 sharex=True, squeeze=False, constrained_layout=True)
        for channel, ax in enumerate(axes[:, 0]):
            ax.plot(sample[:, channel], linewidth=1.0)
            label = str(names[channel]) if names is not None and channel < len(names) else f'Channel {channel}'
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[0, 0].set_title(f'Generated time series — sample {index}')
        axes[-1, 0].set_xlabel('Time index')
        fig.savefig(directory / f'sample_{index:04d}.png', dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='Output .npy; waveforms have shape [N, time, channels].')
    p.add_argument('--metadata', help='Training NPZ providing normalization/STFT metadata for older checkpoints.')
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--solver', choices=['heun', 'euler'], default=None)
    p.add_argument('--channel-id', type=int, default=0)
    p.add_argument('--max-plots', type=int, default=4, help='Number of generated sequences to plot; 0 saves only NPY.')
    args = p.parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or (args.steps is not None and args.steps < 1):
        p.error('Sample count, batch size, and sampling steps must be positive.')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = config_from_dict(ck['config'])
    model = DirectDiffModel(cfg)
    model.load_state_dict(ck['model'], strict=True)
    model.to(device).eval()
    meta = load_preprocessing_metadata(args.metadata) if args.metadata else ck.get('preprocessing', {})
    if args.max_plots < 0:
        p.error('--max-plots must be nonnegative.')
    if not meta:
        p.error('Checkpoint has no reconstruction metadata. Supply --metadata with the training NPZ.')
    output = Path(args.out)
    if output.suffix != '.npy':
        p.error('--out must end in .npy')
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with torch.inference_mode():
        for start in range(0, args.num_samples, args.batch_size):
            count = min(args.batch_size, args.num_samples - start)
            channel_ids = None
            if cfg.num_condition_channels > 0 and cfg.token_mode != 'channel':
                channel_ids = torch.full((count,), args.channel_id, dtype=torch.long, device=device)
            specs = model.generate(count, device=device, sample_steps=args.steps, solver=args.solver, channel_ids=channel_ids)
            value = restore_waveform(specs, meta)
            if not torch.isfinite(value).all():
                raise RuntimeError('Generated output contains non-finite values.')
            chunks.append(value.cpu().numpy())
    result = np.concatenate(chunks)
    np.save(output, result)
    print(f'Saved {result.shape} to {output}')
    if args.max_plots:
        save_timeseries_plots(result[:args.max_plots], output.parent / (output.stem + '_plots'), meta.get('columns'))

if __name__ == '__main__':
    main()
