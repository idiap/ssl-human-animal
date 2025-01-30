import os

import numpy as np
import torch
import torch.nn as nn


class Wav2Vec2(nn.Module):
    """Compute wav2vec2 features of a signal."""

    def __init__(self, name="wav2vec2", feats_dir=None, layer_id=None, nan_rows=None):
        super(Wav2Vec2, self).__init__()

        self.name = name
        self.feats_dir = feats_dir
        self.layer_id = layer_id
        self.nan_rows = nan_rows
        self.feature_dict = dict(np.load(self.feats_dir))
        self.feature_dict = {int(k): v for k, v in self.feature_dict.items()}

        # Replace unwanted row values with the previous ones for standardization.
        # Will lower the performance by a tiny margin
        if self.nan_rows is not None:
            self.nan_rows = np.load(self.nan_rows).astype(int)
            for i in self.nan_rows:
                self.feature_dict[i] = self.feature_dict[i - 1]

    def forward(self, index):
        if isinstance(self.layer_id, int) and self.layer_id in range(self.feature_dict[index].shape[0]):
            return torch.FloatTensor(self.feature_dict[index][self.layer_id]).unsqueeze(0)
        elif self.layer_id == "all":
            return torch.FloatTensor(np.mean(self.feature_dict[index], axis=0)).unsqueeze(0)
        else:
            raise ValueError("Invalid layer ID")
