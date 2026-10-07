from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramSequenceDataset(Dataset):
    """Loads spectrogram sequences shaped [N, T, C, F, W]."""

    def __init__(self, path: str | Path, key: str = "spectrograms") -> None:
        path = Path(path)
        self.raw_sequences: torch.Tensor | None = None
        self.sample_record_ids: np.ndarray | None = None
        self.sample_channel_ids: np.ndarray | None = None
        self.spec_mean: torch.Tensor | None = None
        self.spec_std: torch.Tensor | None = None
        self.metadata: dict[str, int | str] = {}
        if path.suffix == ".npz":
            payload = np.load(path, allow_pickle=True)
            lazy = "spectrogram_storage" in payload and str(payload["spectrogram_storage"]) == "on_demand"
            array = None if lazy else payload[key]
            if "raw_sequences" in payload:
                self.raw_sequences = torch.from_numpy(payload["raw_sequences"]).float()
            if "sample_record_ids" in payload:
                self.sample_record_ids = np.asarray(payload["sample_record_ids"]).astype(str)
            if "sample_channel_ids" in payload:
                self.sample_channel_ids = np.asarray(payload["sample_channel_ids"]).astype(np.int64)
            if "spec_mean" in payload and "spec_std" in payload:
                self.spec_mean = torch.from_numpy(payload["spec_mean"]).float()
                self.spec_std = torch.from_numpy(payload["spec_std"]).float()
            for meta_key in (
                "representation",
                "spectrogram_layout",
                "spec_normalization",
                "original_channels",
                "series_length",
                "window_length",
                "patch_time_bins",
                "freq_bins",
                "n_fft",
                "hop_length",
                "num_condition_channels",
            ):
                if meta_key in payload:
                    value = payload[meta_key]
                    self.metadata[meta_key] = (
                        str(value)
                        if meta_key in {"representation", "spectrogram_layout", "spec_normalization"}
                        else int(value)
                    )
        elif path.suffix == ".npy":
            array = np.load(path)
        else:
            raise ValueError(f"Unsupported data file: {path}")
        if array is None:
            from .lazy_stft import LazySpectrogramArray
            if self.raw_sequences is None or self.spec_mean is None or self.spec_std is None:
                raise ValueError("On-demand STFT requires raw_sequences and spectrum normalization statistics.")
            self.data = LazySpectrogramArray(self.raw_sequences, self.spec_mean, self.spec_std, int(self.metadata.get("n_fft", 126)))
            return
        if array.ndim != 5:
            raise ValueError(f"Expected [N,T,C,F,W], got shape {array.shape}")
        self.data = torch.from_numpy(array).float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        if self.raw_sequences is not None:
            return self.data[index], self.raw_sequences[index]
        return self.data[index]


class SpectrogramPairDataset(Dataset):
    """Builds sampling pairs (A, B) from spectrograms shaped [N, T, C, F, W].

    Pair mode encodes common(A, B) = (A + B)/2 and learns the
    target correction B - common(A, B). Pairs are built before batch shuffling.
    """

    def __init__(
        self,
        base: SpectrogramSequenceDataset,
        pair_gap: int = 1,
        pair_mode: str = "random_same_record",
        pair_seed: int = 7,
    ) -> None:
        if pair_gap < 1:
            raise ValueError("pair_gap must be >= 1")
        if len(base) <= pair_gap:
            raise ValueError(f"Need at least {pair_gap + 1} samples to build pairs.")
        if pair_mode not in {"self", "adjacent", "random", "random_same_record"}:
            raise ValueError("pair_mode must be self, adjacent, random, or random_same_record")
        self.base = base
        self.pair_gap = pair_gap
        self.pair_mode = pair_mode
        self.pair_seed = pair_seed
        self.data = base.data
        self.raw_sequences = base.raw_sequences
        self.spec_mean = base.spec_mean
        self.spec_std = base.spec_std
        self.metadata = base.metadata
        self.sample_channel_ids = base.sample_channel_ids
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> list[tuple[int, int]]:
        if self.pair_mode == "self":
            return [(idx, idx) for idx in range(len(self.base))]
        if self.pair_mode == "adjacent":
            group_ids = self.base.sample_record_ids
            if group_ids is None and self.base.sample_channel_ids is not None:
                group_ids = self.base.sample_channel_ids.astype(str)
            if group_ids is not None:
                pairs = []
                for idx in range(len(self.base) - self.pair_gap):
                    if group_ids[idx] == group_ids[idx + self.pair_gap]:
                        pairs.append((idx, idx + self.pair_gap))
                if not pairs:
                    raise ValueError("No adjacent pairs were created within the same record/channel group.")
                return pairs
            return [(idx, idx + self.pair_gap) for idx in range(len(self.base) - self.pair_gap)]

        rng = np.random.default_rng(self.pair_seed)
        pairs: list[tuple[int, int]] = []
        all_indices = np.arange(len(self.base))

        group_ids = self.base.sample_record_ids
        if group_ids is None and self.base.sample_channel_ids is not None:
            group_ids = self.base.sample_channel_ids.astype(str)

        if self.pair_mode == "random_same_record" and group_ids is not None:
            record_to_indices: dict[str, np.ndarray] = {}
            for record_id in np.unique(group_ids):
                record_to_indices[str(record_id)] = np.flatnonzero(group_ids == record_id)
            for source_idx, record_id in enumerate(group_ids):
                candidates = record_to_indices[str(record_id)]
                candidates = candidates[candidates != source_idx]
                if candidates.size == 0:
                    candidates = all_indices[all_indices != source_idx]
                target_idx = int(rng.choice(candidates))
                pairs.append((source_idx, target_idx))
            return pairs

        for source_idx in range(len(self.base)):
            candidates = all_indices[all_indices != source_idx]
            target_idx = int(rng.choice(candidates))
            pairs.append((source_idx, target_idx))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx, target_idx = self.pairs[index]
        source = self.base.data[source_idx]
        target = self.base.data[target_idx]
        item = {
            "source": source,
            "target": target,
        }
        if self.base.raw_sequences is not None:
            item["source_raw"] = self.base.raw_sequences[source_idx]
            item["target_raw"] = self.base.raw_sequences[target_idx]
        if self.base.sample_channel_ids is not None:
            item["channel_id"] = torch.tensor(int(self.base.sample_channel_ids[target_idx]), dtype=torch.long)
        return item
