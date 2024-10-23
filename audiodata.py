"""
Copyright (c) 2024 by Telecom-Paris
Authoried by Xiaoyu BIE (xiaoyu.bie@telecom-paris.fr)
License agreement in LICENSE.txt
"""

import math
import numpy as np
import pandas as pd
import julius

import torch
import torchaudio
from torch.utils.data import Dataset

class VolumeNorm:
    """
    Volume normalization to a specific loudness [LUFS standard]
    """
    def __init__(self, sample_rate=16000):
        self.lufs_meter = torchaudio.transforms.Loudness(sample_rate)

    def __call__(self, signal, target_loudness=-30, var=0, return_gain=False):
        """
        signal: torch.Tensor [B, channels, L]
        """
        bs = signal.shape[0]
        # LUFS diff
        lufs_ref = self.lufs_meter(signal)
        lufs_target = (target_loudness + (torch.rand(bs) * 2 - 1) * var).to(lufs_ref.device)
        # db to gain
        gain = torch.exp((lufs_target - lufs_ref) * np.log(10) / 20) 
        gain[gain.isnan()] = 0 # zero gain for silent audio
        # norm
        signal *= gain[:, None, None]

        if return_gain:
            return signal, gain
        else:
            return signal
        

class DatasetAudioTrain(Dataset):
    def __init__(self,
        csv_file: str,
        sample_rate: int,
        chunk_size: float = 2.0,
        n_examples: int = 50000,
        trim_silence: bool = True,
        normalize: bool = True,
        lufs_norm_db: float = -27.0,
        lufs_var: float = 2.0,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.csv_files = csv_file
        self.sample_rate = sample_rate # target sampling rate
        self.length = n_examples
        self.chunk_size = chunk_size # negative for entire sentence
        self.trim_silence = trim_silence
        self.normalize = normalize

        # normalize
        if self.normalize:
            self.volume_norm = VolumeNorm(sample_rate=sample_rate)
            self.lufs_norm_db = lufs_norm_db
            self.lufs_var = lufs_var

        # check valid samples
        self.resample_pool = dict()
        orig_utt, orig_len, drop_utt, drop_len = 0, 0, 0, 0
        print('Dataset preparing...')

        metadata = pd.read_csv(self.csv_files)
        if self.trim_silence:
            wav_lens = (metadata['end'] - metadata['start']) / metadata['sr']
        else:
            wav_lens = metadata['length'] / metadata['sr']
        # remove wav files that too short
        orig_utt += len(metadata)
        drop_rows = []
        for row_idx in range(len(wav_lens)):
            orig_len += wav_lens[row_idx]
            if wav_lens[row_idx] < self.chunk_size:
                drop_rows.append(row_idx)
                drop_utt += 1
                drop_len += wav_lens[row_idx]
            else:
                # prepare julius resample
                sr = int(metadata.at[row_idx, 'sr'])
                if sr not in self.resample_pool.keys():
                    old_sr = sr
                    new_sr = self.sample_rate
                    gcd = math.gcd(old_sr, new_sr)
                    old_sr = old_sr // gcd
                    new_sr = new_sr // gcd
                    self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

        metadata = metadata.drop(drop_rows)
        self.metadata = metadata
            
        print("Drop {}/{} utterances ({:.2f}/{:.2f} mins), shorter than {:.2f}s".format(
            drop_utt, orig_utt, drop_len / 60, orig_len / 60, self.chunk_size
        ))
        print('Used utterances, ({:.2f} mins)'.format(
            (orig_len-drop_len) / 60
        ))
        print('Resample pool: {}'.format(list(self.resample_pool.keys())))


    def __len__(self):
        return self.length # can be any number


    def __getitem__(self, idx:int):
        
        idx = np.random.randint(len(self.metadata))
        wav_info = self.metadata.iloc[idx]

        chunk_len = int(wav_info['sr'] * self.chunk_size)

        # slice wav files
        if self.trim_silence: 
            start = np.random.randint(int(wav_info['start']), int(wav_info['end']) - chunk_len + 1)
        else:
            start = np.random.randint(0, int(wav_info['length']) - chunk_len + 1)

        # load file
        x, sr = torchaudio.load(wav_info['filepath'],
                                frame_offset=start,
                                num_frames=chunk_len)

        # single channel
        x = x.mean(dim=0, keepdim=True)

        # resample
        if sr != self.sample_rate:
            x = self.resample_pool[sr](x)

        # normalize
        if self.normalize:
            x = self.volume_norm(signal=x[None,...],
                                target_loudness=self.lufs_norm_db,
                                var=self.lufs_var)[0]

        return x