from typing import Union, Optional
from pathlib import Path

import pandas as pd

from sklearn.model_selection import KFold

DATASETS = ['Muscle', 'pc3', 'HEK']
def prepare_te_and_el_data(
    data_root: Optional[Union[Path, str]],
    kfold_splits: int = 10
):
    data_root = Path(data_root)

    for dataset in DATASETS:
        dataset_path = data_root / (dataset + '_sequence.csv')
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_path} not found.")
        data = pd.read_csv(dataset_path)
        kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
        # save indeces for each fold
        train_ids = []
        test_ids = []
        for (train_idx, test_idx) in kfold.split(data):
            train_ids.append(train_idx)
            test_ids.append(test_idx)

        for i, (train_idx, test_idx) in enumerate(zip(train_ids, test_ids)):
            # Split the data into train, valid and test
            valid_idx = test_ids[i-1]
            # train_idx minus valid_idx
            train_idx = list(set(train_idx) - set(valid_idx))

            train = data.iloc[train_idx]
            valid = data.iloc[valid_idx]
            test = data.iloc[test_idx]
            assert len(train) + len(valid) + len(test) == len(data)
            
            Path(data_root / f'{dataset}/fold{i}').mkdir(parents=True, exist_ok=True)
            train.to_csv(data_root / f'{dataset}/fold{i}/train.csv')
            valid.to_csv(data_root / f'{dataset}/fold{i}/valid.csv')
            test.to_csv(data_root / f'{dataset}/fold{i}/test.csv')
