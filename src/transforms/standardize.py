import os

import numpy as np
import torch
import torch.nn as nn


class Standardize(nn.Module):
    """Standardize features of a signal."""

    def __init__(self, name="standardize", pkl_dir=None, eps= 1e-10):
        super(Standardize, self).__init__()
        self.name = name
        self.pkl_dir = pkl_dir
        self.eps = eps
        self.standard_pkl = dict(np.load(pkl_dir))
        self.means = torch.FloatTensor(self.standard_pkl["means"]).squeeze()
        self.stds = torch.FloatTensor(self.standard_pkl["stds"]).squeeze()

    def forward(self, sample):
        return (sample - self.means) / (self.eps + self.stds)
