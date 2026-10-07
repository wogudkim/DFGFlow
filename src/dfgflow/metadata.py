"""Portable reconstruction metadata, without training samples."""
from pathlib import Path
import numpy as np

KEYS = ('raw_mean', 'raw_std', 'spec_mean', 'spec_std', 'columns',
        'representation', 'original_channels', 'spectrogram_layout',
        'window_length', 'patch_time_bins', 'freq_bins', 'n_fft', 'hop_length',
        'series_length', 'spec_normalization', 'normalization')

def load_preprocessing_metadata(path: str | Path) -> dict:
    if Path(path).suffix != '.npz':
        return {}
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key].tolist() for key in KEYS if key in payload}
