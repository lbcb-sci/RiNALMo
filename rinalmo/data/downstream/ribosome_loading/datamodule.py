from torch.utils.data import DataLoader

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.ribosome_loading.dataset import RibosomeLoadingDataset
from rinalmo.utils.download import download_ribosome_loading_data

VARYING_LEN_25_TO_100_CSV = "GSM4084997_varying_length_25to100.csv.gz"

class RibosomeLoadingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = skip_data_preparation

    def prepare_data(self):
        if not self.skip_data_preparation and not self._data_prepared:
            download_ribosome_loading_data(self.data_root)
            self._data_prepared = True

    def setup(self, stage: Optional[str] = None):
        dataset = RibosomeLoadingDataset(self.data_root / VARYING_LEN_25_TO_100_CSV, alphabet=self.alphabet)
        self.train_dataset, self.random7600_dataset, self.human7600_dataset = dataset.train_eval_split(num_eval_samples_per_len=100)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.random7600_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
                self.human7600_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
        )
