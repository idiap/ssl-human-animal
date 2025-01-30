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


class AbzalievDataset(Dataset):
    """
    This dataloader is reads the AbzalievDataset dataset,
    and constructs a PyTorch DataLoader.
    """

    def __init__(
        self,
        data_dir=None,
        name=None,
        transformation=None,
        standardization=None,
        target_sample_rate=None,
        individual_to_index=None,
        breed_to_index=None,
        individual_to_gender=None,
        gender_to_binary=None,
        calltype_to_index=None,
        individual_to_breed=None,
        typos_individual=None,
        selected_labels=None,
        lengths=None,
    ):
        logger.info(f"Initializing AbzalievDataset dataset.")

        # Variables
        self.data_dir = data_dir
        self.name = name
        self.transformation = transformation
        self.standardization = standardization
        self.target_sample_rate = target_sample_rate
        self.individual_to_index = individual_to_index
        self.calltype_to_index = calltype_to_index
        self.selected_labels = selected_labels
        self.gender_to_binary = gender_to_binary
        self.breed_to_index = breed_to_index
        self.individual_to_gender = individual_to_gender
        self.individual_to_breed = individual_to_breed
        self.typos_individual = typos_individual
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
        calltype_label = self._get_audio_sample_calltype_label(index)
        individual_label = self._get_audio_sample_individual_label(index)
        gender_label = self._get_audio_sample_gender_label(index)
        breed_label = self._get_audio_sample_breed_label(index)

        # Label to return based on selection - in this case we only have one label
        labels = {
            "calltype": calltype_label,
            "caller": individual_label,
            "gender": gender_label,
            "breed": breed_label,
        }
        return_label = labels[self.selected_labels]

        # Return
        return signal, return_label, index

    def _construct_dataframe(self):

        # Construct dataframe
        files = glob.glob(os.path.join(self.data_dir, "data", "*", "*", "*.wav"))
        df = pd.DataFrame(files, columns=["audio_file"])
        df["individual"] = df["audio_file"].apply(lambda x: x.split("/")[-3].split("_")[0].lower())
        df.individual.replace(self.typos_individual, inplace=True)

        df["calltype"] = df["audio_file"].apply(lambda x: x.split("/")[-2])
        df["gender"] = df["individual"].apply(lambda x: self.individual_to_gender[x])
        df["breed"] = df["individual"].apply(lambda x: self.individual_to_breed[x])

        # Convert to IDs
        df["individualID"] = df["individual"].apply(lambda x: self.individual_to_index[x])
        df["calltypeID"] = df["calltype"].apply(lambda x: self.calltype_to_index[x])
        df["breedID"] = df["breed"].apply(lambda x: self.breed_to_index[x])
        df["genderID"] = df["gender"].apply(lambda x: self.gender_to_binary[x]) 

        # Drop the rows where the calltype is S
        df = df[df.calltype != "S"]

        # Start and end
        df["start"] = 0.0
        df["end"] = df["audio_file"].apply(
            lambda x: librosa.get_duration(path=x, sr=self.target_sample_rate)
        )
        df["duration"] = df.end - df.start

        # Re-order columns for cosmetic purposes
        ordered_cols = [
            "audio_file",
            "start",
            "end",
            "duration",
            "individualID",
            "genderID",
            "breedID",
            "calltypeID",
        ]

        df = df[ordered_cols]

        # Reset index to account for dropped rows
        df.reset_index(drop=True, inplace=True)

       # Make a unique vocalization column (across all species)
        df["vocID"] = df.index

        # Convert relevant columns to int
        df.individualID = df.individualID.astype(int)
        df.calltypeID = df.calltypeID.astype(int)
        df.breedID = df.breedID.astype(int)

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
    
    def _get_audio_sample_calltype_label(self, index):
        return self.filelist.calltypeID.iloc[index]

    def _get_audio_sample_individual_label(self, index):
        return self.filelist.individualID.iloc[index]

    def _get_audio_sample_breed_label(self, index):
        return self.filelist.breedID.iloc[index]
    
    def _get_audio_sample_gender_label(self, index):
        return self.filelist.genderID.iloc[index]


if __name__ == "__main__":
    _ = AbzalievDataset()
