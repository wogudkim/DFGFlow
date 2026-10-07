"""Adapt reference MIMIC windows to on-demand STFT training, without subsampling."""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dfgflow.lazy_stft import raw_to_patches


def load_split(root,split,length):
    rows=list(csv.DictReader((root/f'manifest_{split}.csv').open()))
    if not rows: raise ValueError(f'Empty {root}/{split}')
    raw=np.empty((len(rows),length//64,6,64),np.float32)
    for i,row in enumerate(rows):
        values=np.load(root/split/row['file'],allow_pickle=False)
        if values.shape!=(length,6) or not np.isfinite(values).all():
            raise ValueError(f'Invalid raw window in {split}: index {i}')
        raw[i]=values.reshape(length//64,64,6).transpose(0,2,1)
    return raw,rows


def pack(data_root,out,length):
    root=data_root/f'MIMICIV{length}'
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata['bin_minutes']!=15 or metadata['seed']!=2023: raise ValueError('Unexpected reference preprocessing settings')
    train,train_rows=load_split(root,'train',length)
    mean=train.mean(axis=(0,1,3),dtype=np.float64).astype(np.float32)
    std=np.maximum(train.std(axis=(0,1,3),dtype=np.float64).astype(np.float32),1e-6)
    train-=mean[None,None,:,None];train/=std[None,None,:,None]
    sums=np.zeros(12,np.float64);squares=sums.copy();count=0
    for start in range(0,len(train),32):
        spectra=raw_to_patches(train[start:start+32])
        sums+=spectra.sum(axis=(0,1,3,4),dtype=np.float64)
        squares+=np.square(spectra,dtype=np.float64).sum(axis=(0,1,3,4))
        count+=spectra.shape[0]*spectra.shape[1]*spectra.shape[3]*spectra.shape[4]
    spec_mean=(sums/count).astype(np.float32)
    spec_std=np.maximum(np.sqrt(np.maximum(squares/count-(sums/count)**2,0)),1e-6).astype(np.float32)
    report={'length':length,'raw_reference_metadata':metadata,'normalization':'channel z-score; raw and STFT statistics fitted on train windows only',
        'storage':'Raw windows packed without subsampling; STFT computed on demand; reference .npy files unchanged.','splits':{}}
    groups={}
    for split in ('train','val','test'):
        if split=='train': raw,rows=train,train_rows
        else:
            raw,rows=load_split(root,split,length)
            raw-=mean[None,None,:,None];raw/=std[None,None,:,None]
        ids=np.array([r['source_hash'] for r in rows]);groups[split]=set(ids)
        pairs=int((ids[1:]==ids[:-1]).sum())
        if pairs<128: raise ValueError(f'Only {pairs} within-stay pairs in {split}; inspect before training.')
        path=out/f'mimic_iv_len{length}_{split}.npz'
        np.savez_compressed(path,raw_sequences=raw,spectrogram_storage='on_demand',sample_record_ids=ids,
            sample_start_bins=np.array([int(r['start_bin']) for r in rows]),raw_mean=mean,raw_std=std,spec_mean=spec_mean,spec_std=spec_std,
            columns=np.array(metadata['channels']),representation='complex',spectrogram_layout='full_stft_patches',spec_normalization='channel',
            original_channels=6,series_length=length,window_length=64,patch_time_bins=64,freq_bins=64,n_fft=126,hop_length=1,
            bin_minutes=15,split=split,reference_preprocessing_seed=2023)
        report['splits'][split]={'windows':len(raw),'records':len(groups[split]),'valid_adjacent_pairs':pairs,'bytes':path.stat().st_size,
            'manifest_sha256':hashlib.sha256((root/f'manifest_{split}.csv').read_bytes()).hexdigest()}
        print(f'Saved length {length} {split}: {len(raw)} windows, {pairs} within-stay pairs',flush=True)
    assert not groups['train'].intersection(groups['val']|groups['test'])
    assert not groups['val'].intersection(groups['test'])
    (out/f'len{length}_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('data/processed_mimic_reference'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for length in (128,256): pack(a.data_root,a.out,length)
