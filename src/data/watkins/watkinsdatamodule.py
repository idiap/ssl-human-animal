# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import logging
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split

from src import utils

logger = logging.getLogger(__name__)


class WatkinsDataModule(pl.LightningModule):
    """A DataModule wrapped around a PyTorch Dataset."""

    def __init__(
        self,
        data=None,
        batch_size=None,
        train_split=None,
        val_split=None,
        num_workers=None,
    ):
        print(f"Initializing WatkinsDataModule object.")
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load dataset
        print(f"There are {len(self.data)} samples in the dataset.")

        # Split sizes
        train_size = int(self.train_split * len(self.data))
        val_size = int(self.val_split * len(self.data))
        test_size = len(self.data) - train_size - val_size
        splits = [train_size, val_size, test_size]

        # Split datasets
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.data, splits
        )

    def train_dataloader(self):
        # Train Dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # Test dataloader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    _ = WatkinsDataModule()
