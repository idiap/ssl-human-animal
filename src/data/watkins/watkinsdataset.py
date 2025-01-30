# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import glob
import logging
import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WatkinsDataset(Dataset):
    """
    This dataloader is reads the Watkins dataset,
    and constructs a PyTorch DataLoader.
    """

    def __init__(
        self,
        data_dir=None,
        name=None,
        transformation=None,
        standardization=None,
        target_sample_rate=None,
        species_to_index=None,
        selected_labels=None,
        lengths=None,
    ):
        logger.info(f"Initializing WatkinsDataset dataset.")

        # Variables
        self.data_dir = data_dir
        self.name = name
        self.transformation = transformation
        self.standardization = standardization
        self.target_sample_rate = target_sample_rate
        self.species_to_index = species_to_index
        self.selected_labels = selected_labels
        self.lengths = lengths
        if self.lengths:
            self.num_classes = self.lengths[self.selected_labels]

        # Construct
        self.filelist = self._construct_dataframe()
        self.transform_list = [self.transformation]
        self.existing_trans = {
            "catch22",
            "wavlm",
            "byol",
            "pann",
            "wav2vec2",
            "aves_bio",
            "hubert_base",
        }

        if self.transformation in self.existing_trans:
            assert len(list(self.transformation.feature_dict.keys())) == len(
                self.filelist
            )

        if self.standardization:
            self.transform_list.append(self.standardization)
        self.transforms = torch.nn.Sequential(*self.transform_list)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if (
            hasattr(self.transformation, "name")
            and self.transformation.name in self.existing_trans
        ):
            # Get features
            signal = self._transform(index)  # Pre-computed Transformation
        else:
            # Get variables
            audio_sample_path = self._get_audio_sample_path(index)
            signal, sr = self._load_audio_segment(index, audio_sample_path)

            # Preprocess (if necessary)
            signal = self._resample_if_necessary(signal, sr)  # Downsample
            signal = self._mix_down_if_necessary(signal)  # Mono-channel
            signal = self._transform_if_necessary(signal)  # Transformation

        # Get labels
        species_label = self._get_audio_sample_species_label(index)

        # Label to return based on selection - in this case we only have one label
        labels = {"species": species_label}
        return_label = labels[self.selected_labels]

        # Return
        return signal, return_label, index

    def _construct_dataframe(self):

        # Construct dataframe
        df = pd.read_csv(os.path.join(self.data_dir, "data/annotations.csv"))

        # Get vocalization ID from the path
        df["UID"] = df["path"].apply(lambda x: x.split("/")[-1].split(".wav")[0])
        df["audio_file"] = df["path"].apply(
            lambda x: os.path.join(self.data_dir, "data", x)
        )

        # Start will always be 0.0, and end will be length of the audio
        df["start"] = 0.0
        df["end"] = df["audio_file"].apply(
            lambda x: librosa.get_duration(path=x, sr=self.target_sample_rate)
        )
        df["duration"] = df.end - df.start

        # Convert species name to ID
        df["speciesID"] = df["species"].apply(lambda x: self.species_to_index[x])

        # Re-order columns for cosmetic purposes
        ordered_cols = [
            "audio_file",
            "start",
            "end",
            "duration",
            "species",
            "speciesID",
            "UID",
        ]

        df = df[ordered_cols]

        # Make a unique vocalization column (across all species)
        df["vocID"] = df.index

        # Convert relevant columns to int
        df.speciesID = df.speciesID.astype(int)
        df.vocID = df.vocID.astype(int)

        # Return
        return df

    def _load_audio_segment(self, index, audio_sample_path):
        signal, sr = librosa.load(audio_sample_path, sr=None, mono=True)

        # Normalize signal, and put channels first (torchaudio style)
        signal /= np.max(np.abs(signal))
        signal = np.expand_dims(signal, axis=0)
        signal = torch.tensor(signal)
        return signal, sr

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:  # Channels first
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _transform_if_necessary(self, signal):
        if self.transformation:
            signal = self.transformation(signal)
        return signal

    def _transform(self, index):
        return self.transforms(index)

    def _get_audio_sample_path(self, index):
        return self.filelist.audio_file.iloc[index]

    def _get_audio_sample_species_label(self, index):
        return self.filelist.speciesID.iloc[index]


if __name__ == "__main__":
    _ = WatkinsDataset()
