from torch.utils.data import DataLoader

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.secondary_structure.dataset import SecondaryStructureDataset

from rinalmo.utils.download import download_spot_rna_bprna, download_archiveII_fam_splits

# Default train/val/test directory names
TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "valid"
TEST_DIR_NAME = "test"

SUPPORTED_DATASETS = [
    "bpRNA", "archiveII_5s", "archiveII_16s", "archiveII_23s",
    "archiveII_grp1", "archiveII_srp", "archiveII_telomerase", "archiveII_RNaseP",
    "archiveII_tmRNA", "archiveII_tRNA"
]

class SecondaryStructureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        num_workers: int = 0,
        pin_memory: bool = False,
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
        dataset: str = "bpRNA",
        skip_data_preparation: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.alphabet = alphabet

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        if dataset == "bpRNA":
            self.train_dir = f"bpRNA/{TRAIN_DIR_NAME}"
            self.val_dir = f"bpRNA/{VAL_DIR_NAME}"
            self.test_dir = f"bpRNA/{TEST_DIR_NAME}"
        elif dataset in SUPPORTED_DATASETS and dataset.startswith("archiveII"):
            test_rna_family = dataset.split("_")[-1]

            self.train_dir = f"archiveII/{test_rna_family}/{TRAIN_DIR_NAME}"
            self.val_dir = f"archiveII/{test_rna_family}/{VAL_DIR_NAME}"
            self.test_dir = f"archiveII/{test_rna_family}/{TEST_DIR_NAME}"
        else:
            raise NotImplementedError(f"Dataset '{dataset}' is currently not supported! Please use one of the following: {SUPPORTED_DATASETS}")

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = skip_data_preparation

    def prepare_data(self):
        if not self.skip_data_preparation and not self._data_prepared:
            download_spot_rna_bprna(
                self.data_root / "bpRNA", train_dir_name=TRAIN_DIR_NAME,
                val_dir_name=VAL_DIR_NAME, test_dir_name=TEST_DIR_NAME
            )
            download_archiveII_fam_splits(
                self.data_root / "archiveII", train_dir_name=TRAIN_DIR_NAME,
                val_dir_name=VAL_DIR_NAME, test_dir_name=TEST_DIR_NAME
            )

            self._data_prepared = True

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = \
            SecondaryStructureDataset(
                self.data_root / self.train_dir,
                self.alphabet,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
            )
        self.val_dataset = \
            SecondaryStructureDataset(
                self.data_root / self.val_dir,
                self.alphabet,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
            )
        self.test_dataset = \
            SecondaryStructureDataset(
                self.data_root / self.test_dir,
                self.alphabet,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
