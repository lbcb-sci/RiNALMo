from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.splice_site_prediction.dataset import SpliceSiteDataset, SpliceSiteTestDataset

# dataset and data preprocessing code available at https://git.unistra.fr/nscalzitti/spliceator.git

DATA_SUBSET = "GS_1"
LIST_SIZE = 400
class SpliceSiteDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ss_type: str,
        species: str,
        dataset_id: str,
        data_root: Optional[Union[Path, str]],
        test_data_root: Optional[Union[Path, str]],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_root = data_root
        self.test_data_root = test_data_root
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_path = Path(self.data_root + f"/{DATA_SUBSET}/{dataset_id}/Train_{ss_type}_{LIST_SIZE}.csv") if data_root else None
        self.val_path = Path(self.data_root + f"/{DATA_SUBSET}/{dataset_id}/Test_{ss_type}_{LIST_SIZE}.csv") if data_root else None

        # test
        self.test_path_data = Path(self.test_data_root + f"/{species}/SA_sequences_{ss_type}_{LIST_SIZE}_Final_3.fasta") if test_data_root else None
        self.test_path_labels = Path(self.test_data_root + f"/{species}/SA_labels_{ss_type}_{LIST_SIZE}_Final_3.fasta") if test_data_root else None
        
    def setup(self, stage: Optional[str] = None):
        if self.data_root :
            self.train_dataset = SpliceSiteDataset(self.train_path, alphabet=self.alphabet)
            self.val_dataset = SpliceSiteDataset(self.val_path, alphabet=self.alphabet)
        
        elif self.test_data_root:
            self.test_dataset = SpliceSiteTestDataset(self.test_path_data, self.test_path_labels, alphabet=self.alphabet)


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
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
