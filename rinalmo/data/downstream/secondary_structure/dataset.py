import torch
from torch.utils.data import Dataset

from typing import Union
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import parse_sec_struct_file

class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        alphabet: Alphabet = Alphabet(),
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
        ss_file_extensions: list[str] = ["ct", "bpseq", "st"],
    ):
        super().__init__()

        self.alphabet = alphabet
        self.data_dir = Path(data_dir)

        # Collect secondary structure file paths
        self.ss_paths = []
        for ss_file_ext in ss_file_extensions:
            for ss_file_path in list(self.data_dir.glob(f"**/*.{ss_file_ext}")):
                seq, _ = parse_sec_struct_file(ss_file_path)

                if len(seq) >= min_seq_len and len(seq) <= max_seq_len:
                    self.ss_paths.append(ss_file_path)

    def __len__(self):
        return len(self.ss_paths)

    def __getitem__(self, idx):
        ss_id = self.ss_paths[idx].stem
        seq, sec_struct = parse_sec_struct_file(self.ss_paths[idx])

        seq_encoded = torch.tensor(self.alphabet.encode(seq), dtype=torch.int64)
        sec_struct = torch.tensor(sec_struct)

        return ss_id, seq, seq_encoded, sec_struct
