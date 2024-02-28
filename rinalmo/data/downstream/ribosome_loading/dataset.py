import torch
from torch.utils.data import Dataset, Subset

import pandas as pd

from typing import Union
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

MIN_EVAL_SEQ_LEN = 25
MAX_EVAL_SEQ_LEN = 100

def _generate_eval_set_eq_sampled_lens(df: pd.DataFrame, min_seq_len: int, max_seq_len: int, num_samples_per_len: int):
    eval_idcs = set()

    for len in range(min_seq_len, max_seq_len + 1):
        len_df = df[df['len'] == len].copy()
        len_df.sort_values('total_reads', inplace=True, ascending=False)
        eval_idcs.update(len_df.iloc[:num_samples_per_len].index.values.tolist())

    return eval_idcs

class RibosomeLoadingDataset(Dataset):
    def __init__(
        self,
        mrl_csv: Union[str, Path],
        alphabet: Alphabet,
        pad_to_max_len: bool = True,
    ):
        super().__init__()

        self.df = pd.read_csv(mrl_csv)
        self.df.dropna(subset=['rl'], inplace=True) # Remove entries with missing ribosome loading value

        self.alphabet = alphabet

        self.max_enc_seq_len = -1
        if pad_to_max_len:
            self.max_enc_seq_len = self.df['utr'].str.len().max() + 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]

        seq = df_row['utr']
        seq_encoded = torch.tensor(self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), dtype=torch.long)

        rl = torch.tensor(df_row['rl'], dtype=torch.float32)

        return seq_encoded, rl
    
    def train_eval_split(self, num_eval_samples_per_len: int = 100):
        assert 'set' in self.df.columns and 'total_reads' in self.df.columns, "Given CSV file cannot be split into training and validation sets!"

        self.df = self.df[(self.df['set'] == 'random') | (self.df['set'] == 'human')]
        self.df = self.df[self.df['total_reads'] >= 10] # TODO: Other test sets don't have this column?
        self.df.drop_duplicates('utr', inplace=True, keep=False)

        self.df.reset_index(inplace=True)

        random7600_idcs = _generate_eval_set_eq_sampled_lens(
            self.df[self.df['set'] == 'random'],
            min_seq_len=MIN_EVAL_SEQ_LEN,
            max_seq_len=MAX_EVAL_SEQ_LEN,
            num_samples_per_len=num_eval_samples_per_len
        )
        random7600_ds = Subset(self, indices=list(random7600_idcs))

        human7600_idcs = _generate_eval_set_eq_sampled_lens(
            self.df[self.df['set'] == 'human'],
            min_seq_len=MIN_EVAL_SEQ_LEN,
            max_seq_len=MAX_EVAL_SEQ_LEN,
            num_samples_per_len=num_eval_samples_per_len
        )
        human7600_ds = Subset(self, indices=list(human7600_idcs))

        random_df = self.df[self.df['set'] == 'random']
        train_idcs = set(random_df.index.values.tolist()) - random7600_idcs
        train_ds = Subset(self, indices=list(train_idcs))

        return train_ds, random7600_ds, human7600_ds
