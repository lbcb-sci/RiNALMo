import torch
import torch.nn as nn

import numpy as np

# Inspired by https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
class StandardScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer("_mean", torch.tensor([0.0]))
        self.register_buffer("_std", torch.tensor([1.0]))

        self._seen_samples = []

        self._need_update = False

    def _update_mean_and_std(self) -> None:
        self._mean[0] = np.mean(self._seen_samples)
        self._std[0] = np.std(self._seen_samples)

    def partial_fit(self, x: torch.Tensor) -> None:
        self._need_update = True
        self._seen_samples.extend(x.cpu().view(-1).tolist())

        self._update_mean_and_std()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def inverse_transform(self, scaled_x: torch.Tensor) -> torch.Tensor:
        return scaled_x * self._std + self._mean
