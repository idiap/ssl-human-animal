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


class MarmosetDataModule(pl.LightningModule):
    def __init__(
        self,
        data,
        batch_size,
        train_split,
        val_split,
        num_workers,
    ):
        logger.info(f"Initializing DataModule.")
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data = data
        self.hparams.batch_size = batch_size
        self.hparams.train_split = train_split
        self.hparams.val_split = val_split
        self.hparams.num_workers = num_workers

    def setup(self, stage=None):
        # Load dataset
        print(f"There are {len(self.hparams.data)} samples in the dataset.")

        # Split sizes
        train_size = int(self.hparams.train_split * len(self.hparams.data))
        val_size = int(self.hparams.val_split * len(self.hparams.data))
        test_size = len(self.hparams.data) - train_size - val_size
        splits = [train_size, val_size, test_size]

        # Split datasets
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.hparams.data, splits, generator=seed
        )

        # If we're using handcrafted features, standardize
        # self.train_dataset

    def train_dataloader(self):
        # Train Dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        # Validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        # Test datalTupleoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MarmosetDataModule()
