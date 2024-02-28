import torch
from torch.utils.data import Dataset, Subset

import pandas as pd
import numpy as np

from typing import Union
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

from Bio import SeqIO


class SpliceSiteDataset(Dataset):
    def __init__(
        self,
        ss_csv: Union[str, Path],
        alphabet: Alphabet,
        pad_to_max_len: bool = True
    ):
        super().__init__()

        self.df = pd.read_csv(ss_csv, delimiter=';', header=None)

        self.alphabet = alphabet

        if pad_to_max_len:
            self.max_enc_seq_len = self.df[1].str.len().max() + 2

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        seq = df_row[1]
        seq_encoded = torch.tensor(self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), dtype=torch.long)

        label = torch.tensor(df_row[2], dtype=torch.double)

        return seq_encoded, label


class SpliceSiteTestDataset(Dataset):
    def __init__(
        self,
        seq_data: Union[str, Path],
        label_data: Union[str, Path],
        alphabet: Alphabet,
        pad_to_max_len: bool = True
    ):
        super().__init__()

        self.sequences = np.loadtxt(seq_data, dtype='str')
        self.labels = np.loadtxt(label_data, dtype='int')

        self.alphabet = alphabet

        if pad_to_max_len:
            self.max_enc_seq_len = len(self.sequences[0][:]) + 2

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx][:]
        seq_encoded = torch.tensor(self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.double)

        return seq_encoded, label