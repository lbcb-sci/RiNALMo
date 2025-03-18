import torch
from torch.utils.data import Dataset

import pandas as pd

from typing import Union
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

class ExpressionLevelDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[Path, str], 
        alphabet: Alphabet
    ):
        self.data = pd.read_csv(csv_path)
        self.alphabet = alphabet

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx]['utr'].replace('<pad>', '')
        el_target = self.data.iloc[idx]['rnaseq_log']
        
        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.int64)

        return tokens, el_target
