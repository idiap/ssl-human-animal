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


class IMVDataModule(pl.LightningModule):
    def __init__(
        self,
        dataset=None,
        batch_size=None,
        audio_dir=None,
        labels_file=None,
        transformation=None,
        sample_rate=None,
        calltype_to_index=None,
        selected_labels=None,
        lengths=None,
    ):

        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.audio_dir = audio_dir
        self.labels_file = labels_file
        self.transformation = transformation
        self.target_sample_rate = sample_rate
        self.calltype_to_index = calltype_to_index
        self.selected_labels = selected_labels
        self.lengths = lengths
        if self.lengths:
            self.num_classes = self.lengths[self.selected_labels]

    def setup(self, stage):
        # Load dataset
        data = self.dataset(
            audio_dir=self.audio_dir,
            labels_file=self.labels_file,
            transformation=self.transformation,
            target_sample_rate=self.target_sample_rate,
            calltype_to_index=self.calltype_to_index,
            labels_of_choice=self.labels_of_choice,
        )
        print(f"There are {len(data)} samples in the dataset.")

        # Apply transforms eg normlization

        # Split sizes
        train_size = int(0.7 * len(data))
        val_size = int(0.2 * len(data))
        test_size = len(data) - train_size - val_size
        splits = [train_size, val_size, test_size]

        # Split datasets
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            data, splits, generator=seed
        )

    def train_dataloader(self):
        # Train Dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn,
            num_workers=7,
        )

    def val_dataloader(self):
        # Validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=7,
        )

    def test_dataloader(self):
        # Test dataloader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=7,
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
    _ = IMVDataModule()
