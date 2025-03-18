import torch
from torch.utils.data import Dataset

import pandas as pd

from typing import Union
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

class ncRNADataset(Dataset):
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
        seq = self.data.iloc[idx]['sequence']
        fam = self.data.iloc[idx]['name']
        class_id = self.data.iloc[idx]['class_id']
        
        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.int64)

        return fam, tokens, class_id
