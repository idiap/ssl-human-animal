# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

from torch import nn

class CNN_16KHz_Marmosets(nn.Module):
    def __init__(
        self,
        selected_labels: str,
        lengths: dict,
        num_input: int = 1,
        win_size_s: float = 0.03,
        win_shift_s: float = 0.005,
        sample_rate: int = 16000,
        num_channels: int = 128,
        flatten_size: int = 4,
    ):
        super().__init__()
        self.num_input = num_input
        self.win_size_s = win_size_s
        self.win_shift_s = win_shift_s
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.flatten_size = flatten_size
        self.selected_labels = selected_labels
        self.lengths = lengths
        self.num_classes = self.lengths[self.selected_labels]

        kernel_size = int(self.win_size_s * self.sample_rate)
        stride = int(self.win_shift_s * self.sample_rate)

        self.conv1 = nn.Conv1d(
            in_channels=self.num_input,
            out_channels=self.num_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=self.num_channels * 2,
            kernel_size=10,
            stride=5,
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=2 * self.num_channels,
            out_channels=self.num_channels * 2 * 2,
            kernel_size=1,
            stride=1,
            padding=1,
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(
            in_channels=2 * 2 * self.num_channels,
            out_channels=self.num_channels * 2 * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu4 = nn.ReLU()

        self.adapt = nn.AdaptiveAvgPool1d(self.flatten_size)  # <-- Arbitrary
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=self.flatten_size * 2 * 2 * self.num_channels,
            out_features=2 * 2 * self.num_channels,
        )
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(
            in_features=2 * 2 * self.num_channels,
            out_features=2 * self.num_channels,
        )
        self.relu6 = nn.ReLU()

        self.fc = nn.Linear(
            in_features=2 * self.num_channels, out_features=self.num_classes
        )

        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(out_features=10)
        # self.relu5 = nn.ReLU()
        # self.fc = nn.Linear(10, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.adapt(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    _ = CNN_16KHz_Marmosets()
