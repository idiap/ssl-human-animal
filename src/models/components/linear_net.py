# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(
        self,
        selected_labels: str,
        lengths: dict,
        num_input: int,
    ):
        super(LinearNet, self).__init__()
        self.selected_labels = selected_labels
        self.lengths = lengths
        self.num_classes = self.lengths[self.selected_labels]

        # Model
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_input, self.num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    _ = LinearNet()
