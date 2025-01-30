import os

import numpy as np
import torch
import torch.nn as nn


class WavLM(nn.Module):
    """Compute wavlm features of a signal."""

    def __init__(self, name="wavlm", feats_dir=None, layer_id=None, nan_rows=None):
        super(WavLM, self).__init__()

        self.name = name
        self.feats_dir = feats_dir
        self.layer_id = layer_id
        self.nan_rows = nan_rows
        self.feature_dict = dict(np.load(self.feats_dir))
        self.feature_dict = {int(k): v for k, v in self.feature_dict.items()}

        if self.nan_rows is not None:
            self.nan_rows = np.load(self.nan_rows).astype(int)
            self.replace_nan_values()

    def replace_nan_values(self):
        for i in self.nan_rows:
            if i in self.feature_dict:
                # Find the nearest previous non-NaN value
                replacement_found = False
                for j in range(i - 1, -1, -1):
                    if j in self.feature_dict and not np.isnan(self.feature_dict[j]).any():
                        self.feature_dict[i] = self.feature_dict[j]
                        replacement_found = True
                        break
                if not replacement_found:
                    print(f"Warning: No non-NaN replacement found for index {i}")
            else:
                print(f"Warning: Index {i} not found in feature_dict")

    def forward(self, index):
        return torch.FloatTensor(self.feature_dict[index][self.layer_id]).unsqueeze(0)
