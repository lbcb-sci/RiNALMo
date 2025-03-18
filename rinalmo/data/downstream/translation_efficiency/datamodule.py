from torch.utils.data import DataLoader
import torch

from sklearn.model_selection import KFold

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import *
from rinalmo.data.downstream.translation_efficiency.dataset import TranslationEffDataset
from rinalmo.utils.download import download_mrna_te_and_el_data
from rinalmo.utils.prepare_mrna_te_and_el_data import prepare_te_and_el_data

class TranslationEffDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Optional[Union[Path, str]],
        cell_line: str = 'Muscle',
        fold: int = 0,
        alphabet: Alphabet = Alphabet(),
        kfold_splits: int = 10,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation = True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.cell_line = cell_line
        self.fold = fold

        self.alphabet = alphabet

        self.kfold_splits = kfold_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = False

    def prepare_data(self):
        if not self.skip_data_preparation and not self._data_prepared:
            print(f"Downloading the data to {self.data_root} ...")
            download_mrna_te_and_el_data(self.data_root)
            print(f"Downloaded data!")
            print(f"Preparing data...")
            prepare_te_and_el_data(self.data_root, self.kfold_splits)
            print(f'Data prepared!')
            self._data_prepared = True

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TranslationEffDataset(self.data_root / (f'{self.cell_line}/fold{self.fold}/train.csv'), self.alphabet)
        self.val_dataset = TranslationEffDataset(self.data_root / (f'{self.cell_line}/fold{self.fold}/valid.csv'), self.alphabet)
        self.test_dataset = TranslationEffDataset(self.data_root / (f'{self.cell_line}/fold{self.fold}/test.csv'), self.alphabet)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=BatchPreparator(self.alphabet.get_idx(PAD_TKN)),
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=BatchPreparator(self.alphabet.get_idx(PAD_TKN))
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=BatchPreparator(self.alphabet.get_idx(PAD_TKN))
        )

class BatchPreparator:
    def __init__(
        self,
        pad_tkn_idx: int
    ):
        super().__init__()

        self.pad_tkn_idx = pad_tkn_idx
    
    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(tokens) for tokens, _ in batch)

        padded_tokens = torch.full((batch_size, max_len), fill_value=self.pad_tkn_idx, dtype=torch.int64)

        for i, (tokens, _) in enumerate(batch):
            tokens_len = len(tokens)
            padded_tokens[i, :tokens_len] = tokens

        te_log = torch.tensor([te_log for _, te_log in batch], dtype=torch.float32)

        return padded_tokens, te_log
