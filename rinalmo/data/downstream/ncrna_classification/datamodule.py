from torch.utils.data import DataLoader
import torch

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import *
from rinalmo.data.downstream.ncrna_classification.dataset import ncRNADataset
from rinalmo.utils.download import download_ncrna_data
from rinalmo.utils.prepare_ncrna_classification_data import prepare_ncrna_classification_data, add_noise_to_ncrna_data

class ncRNADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Optional[Union[Path, str]],
        boundary_noise: str = '',
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation = True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.alphabet = alphabet
        self.boundary_noise = boundary_noise

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = False

    def prepare_data(self):
        if not self.skip_data_preparation and not self._data_prepared:
            print(f"Downloading the data to {self.data_root} ...")
            download_ncrna_data(self.data_root)
            print(f"Downloaded data!")
            print(f"Preparing data...")
            prepare_ncrna_classification_data(self.data_root)
            print(f"Noiseless data prepared...")
            add_noise_to_ncrna_data(self.data_root)
            print(f'Data prepared!')
            self._data_prepared = True

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ncRNADataset(self.data_root / ("train" + self.boundary_noise + ".csv"), self.alphabet)
        self.val_dataset = ncRNADataset(self.data_root / ("val" + self.boundary_noise + ".csv"), self.alphabet)
        self.test_dataset = ncRNADataset(self.data_root / ("test" + self.boundary_noise + ".csv"), self.alphabet)

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
        max_len = max(len(tokens) for _, tokens, _ in batch)

        padded_tokens = torch.full((batch_size, max_len), fill_value=self.pad_tkn_idx, dtype=torch.int64)

        for i, (_, tokens, _) in enumerate(batch):
            tokens_len = len(tokens)
            padded_tokens[i, :tokens_len] = tokens

        fams = [fam for fam, _, _ in batch]
        class_ids = torch.tensor([class_id for _, _, class_id in batch], dtype=torch.int64)

        return fams, padded_tokens, class_ids
