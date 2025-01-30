# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

from torch import nn


class DenseNet(nn.Module):
    def __init__(
        self,
        selected_labels: str,
        lengths: dict,
        num_input: int,
        lin1_size: int = 128,
        lin2_size: int = 64,
        lin3_size: int = 32,
    ):
        super().__init__()
        self.selected_labels = selected_labels
        self.lengths = lengths
        self.num_classes = self.lengths[self.selected_labels]

        # Model
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_input, lin1_size)
        self.ln1 = nn.LayerNorm(lin1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(lin1_size, lin2_size)
        self.ln2 = nn.LayerNorm(lin2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(lin2_size, lin3_size)
        self.ln3 = nn.LayerNorm(lin3_size)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(lin3_size, self.num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    _ = DenseNet()
