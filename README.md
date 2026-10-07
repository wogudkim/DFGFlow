# DFGFlow

DFGFlow is a standalone package for time-series preprocessing, autoencoder and flow-matching training, and generation.

## Scope

- Preprocessing for CSV time series (including Energy and ETTh1), Accel XYZ NPY files, and MIMIC-IV
- Model architecture, training losses, and training progress logs
- Complex STFT representations with separate real and imaginary channels, and waveform reconstruction
- Time-series generation and per-channel 1D plots

The repository does not include datasets, pretrained checkpoints, experiment outputs, evaluation metrics, ablation runners, ground-truth comparison plots, or spectrogram visualizations.

## Generated outputs

- `.npy`: Time series in the original data units, with shape **[number of samples, sequence length, number of channels]**.
- `<output_name>_plots/sample_XXXX.png`: One figure per generated sample, with a 1D subplot for each channel. By default, the first four samples are plotted. Setting `--max-plots` to `0` saves only the NPY file.

New checkpoints store normalization statistics and STFT settings without storing training samples. Generation therefore does not require the original training data. Older checkpoints without this metadata require the corresponding training NPZ file to be supplied separately. Waveform generation uses complex representations that preserve phase information.

## Repository structure

- `src/dfgflow/config.py`, `model.py`: Model configuration and architecture
- `src/dfgflow/data.py`, `lazy_stft.py`: Training data loading and on-demand STFT
- `src/dfgflow/losses.py`: Training losses and STFT reconstruction
- `src/dfgflow/metadata.py`: Waveform reconstruction metadata
- `src/dfgflow/train.py`: Autoencoder and flow-matching training
- `src/dfgflow/generate.py`: Time-series NPY output and 1D plots
- `scripts/`: Preprocessing tools

The Python package is named `dfgflow`. Model class names, checkpoint `state_dict` keys, and checkpoint filenames retain their original names for compatibility with DirectDiff-TS. The model architecture and training computations are unchanged.

## Preprocessing notes

The general CSV and Accel preprocessing tools fit normalization statistics on the supplied input, so only training data should be provided when fitting these statistics. The MIMIC packer reuses training-set statistics for validation and test data. Adjacent-pair training requires adjacent windows within the same record.
