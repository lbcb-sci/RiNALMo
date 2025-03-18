from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.splice_site_prediction.dataset import SpliceSiteDataset, SpliceSiteTestDataset
from rinalmo.utils.download import download_splice_site_data

import os
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

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
        skip_data_preparation = True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.test_data_root = Path(test_data_root) if test_data_root else None
        self.species = species
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.ss_type = ss_type
        self.dataset_id = dataset_id

        self.train_path = Path(self.data_root / f"{DATA_SUBSET}/{dataset_id}/Train_{ss_type}_{LIST_SIZE}.csv") if data_root else None
        self.val_path = Path(self.data_root / f"{DATA_SUBSET}/{dataset_id}/Val_{ss_type}_{LIST_SIZE}.csv") if data_root else None

        # test
        self.test_path_data = Path(self.test_data_root / f"{species}/SA_sequences_{ss_type}_{LIST_SIZE}_Final_3.fasta") if test_data_root else None
        self.test_path_labels = Path(self.test_data_root / f"{species}/SA_labels_{ss_type}_{LIST_SIZE}_Final_3.fasta") if test_data_root else None
        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = False
        
    def _merge_csvs(self, pos_paths, neg_paths):
        sequences = []
        groups = []
        labels = []
        for label, files in [[1, pos_paths], [0, neg_paths]]:
            for fn in files:
                bn = os.path.basename(fn)
                with open(fn) as infile:
                    for l in infile:
                        if l.startswith("ID_uniprot"):
                            continue
                        fields = l.strip().split(';')
                        if len(fields[1]) < 100:
                            seq = fields[2]
                        else:
                            seq = fields[1]
                        # assert len(seq) == 600, "{}".format((len(seq), fn, fields))
                        skip_left = (len(seq) - LIST_SIZE) // 2 # + np.random.randint(-10, 11)
                        # if self.shift > 0:
                        #     skip_left += np.random.randint(-self.shift, self.shift + 1)
                        seq = seq[skip_left:skip_left + LIST_SIZE]
                        sequences.append(seq)
                        groups.append(fields[0].split('_')[-1])
                        labels.append(label)
                        # self.samples.append((bn, label, fields[0]))
        labels = np.array(labels)
        groups = np.array(groups)
        sequences = np.array(sequences)
        return groups, sequences, labels
    
    def _save_training_validation_folds(self, groups, sequences, labels, ss_type):
        print("Saving training and validation folds...")

        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020) # seed from splicebert default
        for idx, (train_inds, val_inds) in enumerate(splitter.split(np.arange(len(labels)), y=labels)):
            dataset_id = f"db_{idx + 1}"
            # write to csv, row idx 1 = sequence, row idx 2 = label, row idx 0 is unknown should be some sort of id probably or group
            train_df = pd.DataFrame({
                0: groups[train_inds],  # Adding a group column (not used in the Dataset class but requires something) ex. of group: 'human'
                1: sequences[train_inds],
                2: labels[train_inds]
            })

            val_df = pd.DataFrame({
                0: groups[val_inds],  # Adding a group column (not used in the Dataset class but requires something)
                1: sequences[val_inds],
                2: labels[val_inds]
            })
            
            train_path = Path(self.data_root / f"{DATA_SUBSET}/{dataset_id}/Train_{ss_type}_{LIST_SIZE}.csv")
            val_path = Path(self.data_root / f"{DATA_SUBSET}/{dataset_id}/Val_{ss_type}_{LIST_SIZE}.csv")
            
            os.makedirs(train_path.parent, exist_ok=True)
            os.makedirs(val_path.parent, exist_ok=True)
            train_df.to_csv(train_path, sep=';', header=False, index=False)
            val_df.to_csv(val_path, sep=';', header=False, index=False)

    def _parse_test_txt(self, pos_path_seq, neg_path_seq):
        # read negative txt as csv
        df_neg = pd.read_csv(neg_path_seq, delimiter=';', header=None)
        sequences_neg = df_neg[2].to_numpy(dtype=str)
        labels_neg = np.zeros_like(sequences_neg, dtype=int)

        # read positive
        df_pos = pd.read_csv(pos_path_seq, delimiter=';', header=None)
        sequences_pos = df_pos[2].to_numpy(dtype=str)
        labels_pos = np.ones_like(sequences_pos, dtype=int)

        # merge neg and pos
        sequences = np.concatenate((sequences_neg, sequences_pos))
        labels = np.concatenate((labels_neg, labels_pos))

        return sequences, labels
        
    def prepare_data(self):
        # code adapted from Splicebert: https://github.com/chenkenbio/SpliceBERT/blob/main/examples/04-splicesite-prediction/spliceator_data.py
        # Splicebert paper: https://www.biorxiv.org/content/10.1101/2023.01.31.526427v1
        # Data from: https://zenodo.org/record/7995778/files/data.tar.gz?download=1
        
        if not self.skip_data_preparation and not self._data_prepared:
            if self.data_root is None or self.test_data_root is None:
                raise ValueError("please specify the data root and test data root in order to prepare data")
            
            ss_types = ['acceptor', 'donor']

            print(f"Downloading the data to {self.data_root} ...")
            dataset_dir = download_splice_site_data(self.data_root)
            self._data_prepared = True
            print(f"downloaded data! Training data in: {dataset_dir}")

            print(f"Processing Training data...")
            train_dir = dataset_dir / "Training_data"
            for ss_type in ss_types:
                negative_file_path = train_dir / f"Negative/GS/GS_1/NEG_600_{ss_type}.csv"
                positive_file_path = train_dir / f"Positive/GS/POS_{ss_type}_600.csv"
                neg_paths = [negative_file_path]
                pos_paths = [positive_file_path]

                groups, sequences, labels = self._merge_csvs(pos_paths=pos_paths, neg_paths=neg_paths)
                self._save_training_validation_folds(groups, sequences, labels, ss_type)
            print(f"Finished processing and saving training, validation folds!")

            print("Processing Benchmark (Test) data...")
            benchmarks = ['Danio', 'Fly', 'Thaliana', 'Worm']
            test_dir = dataset_dir / "Benchmarks"
            for ss_type in ss_types:
                for species in benchmarks:
                    bm_pos_path_seq = test_dir / species / f"SA_sequences_{ss_type}_400_Final_3.positive.txt" # fn;POS;sequence
                    bm_neg_path_seq = test_dir / species / f"SA_sequences_{ss_type}_400_Final_3.negative.txt"

                    sequences, labels = self._parse_test_txt(bm_pos_path_seq, bm_neg_path_seq)
                    test_path_data = Path(self.test_data_root / f"{species}/SA_sequences_{ss_type}_{LIST_SIZE}_Final_3.fasta")
                    test_path_labels = Path(self.test_data_root / f"{species}/SA_labels_{ss_type}_{LIST_SIZE}_Final_3.fasta")

                    os.makedirs(test_path_data.parent, exist_ok=True)
                    os.makedirs(test_path_labels.parent, exist_ok=True)
                    np.savetxt(test_path_data, sequences, fmt='%s')
                    np.savetxt(test_path_labels, labels, fmt='%d')

    def setup(self, stage: Optional[str] = None):
        if self.data_root :
            self.train_dataset = SpliceSiteDataset(self.train_path, alphabet=self.alphabet)
            self.val_dataset = SpliceSiteDataset(self.val_path, alphabet=self.alphabet)
        
        if self.test_data_root:
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
