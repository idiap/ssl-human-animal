import os

import numpy as np
import torch
import torch.nn as nn


class HuBERT(nn.Module):
    """Compute wavlm features of a signal."""

    def __init__(
        self, name="hubert_base", feats_dir=None, layer_id=None, nan_rows=None
    ):
        super(HuBERT, self).__init__()

        self.name = name
        self.feats_dir = feats_dir
        self.layer_id = layer_id
        self.nan_rows = nan_rows
        self.feature_dict = dict(np.load(self.feats_dir))
        self.feature_dict = {int(k): v for k, v in self.feature_dict.items()}

    def forward(self, index):
        return torch.FloatTensor(self.feature_dict[index][self.layer_id]).unsqueeze(0)
