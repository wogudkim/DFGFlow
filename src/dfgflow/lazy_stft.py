"""Materialize complex STFT patches on demand from normalized raw windows."""
from __future__ import annotations
import numpy as np
import torch


def raw_to_patches(raw: np.ndarray, n_fft: int = 126) -> np.ndarray:
    """[B,T,C,W] -> [B,T,2C,F,W], symmetric Hann, centered reflect padding."""
    batch, tokens, channels, width = raw.shape
    signal = raw.transpose(0,2,1,3).reshape(batch,channels,tokens*width)
    half=n_fft//2
    padded=np.pad(signal,((0,0),(0,0),(half,n_fft-half-1)),mode='reflect')
    frames=np.lib.stride_tricks.sliding_window_view(padded,n_fft,axis=-1)
    spectrum=np.fft.rfft(frames*np.hanning(n_fft).astype(np.float32),axis=-1)
    parts=np.stack([spectrum.real,spectrum.imag],axis=2).astype(np.float32)
    maps=parts.reshape(batch,2*channels,tokens*width,n_fft//2+1).transpose(0,1,3,2)
    return maps.reshape(batch,2*channels,n_fft//2+1,tokens,width).transpose(0,3,1,2,4).copy()


class LazySpectrogramArray:
    """Tensor-like indexing for existing data loaders without a full STFT cache."""
    def __init__(self, raw, mean, std, n_fft=126):
        self.raw=raw
        self.mean=np.asarray(mean,dtype=np.float32).reshape(1,1,-1,1,1)
        self.std=np.asarray(std,dtype=np.float32).reshape(1,1,-1,1,1)
        self.n_fft=n_fft
        n,t,c,w=raw.shape
        self.shape=(n,t,2*c,n_fft//2+1,w)
        self.ndim=5

    def __len__(self):
        return self.shape[0]

    def __getitem__(self,index):
        selected=self.raw[index]
        single=selected.ndim==3
        if single: selected=selected[None]
        values=raw_to_patches(selected.numpy(),self.n_fft)
        values=(values-self.mean)/self.std
        return torch.from_numpy(values[0] if single else values)
