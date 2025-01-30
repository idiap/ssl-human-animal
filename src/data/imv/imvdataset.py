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


class IMVDataset(Dataset):
    """This dataloader reads the InfantMarmosetsVox
    marmosets data, and constructs a PyTorch DataLoader.
    Paper: https://www.isca-speech.org/archive/interspeech_2023/sarkar23_interspeech.html
    """

    def __init__(
        self,
        data_dir=None,
        name=None,
        transformation=None,
        standardization=None,
        target_sample_rate=None,
        calltype_to_index=None,
        nan_rows=None,
        selected_labels=None,
        lengths=None,
    ):
        logger.info("Initializing InfantMarmosetsVox Dataset.")
        self.data_dir = data_dir
        self.name = name
        self.transformation = transformation
        self.standardization = standardization
        self.target_sample_rate = target_sample_rate
        self.calltype_to_index = calltype_to_index
        self.nan_rows = nan_rows
        self.selected_labels = selected_labels
        self.lengths = lengths
        if self.lengths:
            self.num_classes = self.lengths[self.selected_labels]

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

        # Label to return based on selection
        labels = {
            "calltype": calltype_label,
            "caller": individual_label,
        }
        return_label = labels[self.selected_labels]

        # Return
        return signal, return_label, index

    def _construct_dataframe(self):
        # Vocalizations segments
        labels_file = os.path.join(self.data_dir, "labels.csv")
        df = pd.read_csv(labels_file)
        df["twinID"] = df.filename.apply(lambda x: x.split("_")[1][-1])
        df["individualID"] = df.filename.apply(lambda x: x.split("_")[2][-1])
        df["date"] = df.filename.apply(lambda x: x.split("_")[0])

        # Process
        df.drop(columns=["filename"], inplace=True)
        df.rename(columns={"calltype": "calltypeID"}, inplace=True)
        df.rename(columns={"caller": "callerID"}, inplace=True)
        # df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

        df["audio_file"] = df.apply(
            lambda row: os.path.join(
                self.data_dir,
                "data/twin_" + row.twinID,
                row.date
                + "_Twin"
                + row.twinID
                + "_marmoset"
                + row.individualID
                + ".wav",
            ),
            axis=1,
        )

        # Check all files exist
        df["exists"] = df["audio_file"].astype(str).map(os.path.exists)

        # Drop rows where the audio files don't exist
        df = df[df.exists == True]

        # Drop rows where there are weird CalltypeIDs (only 7 lines)
        df = df[df.calltypeID <= 12]

        # Drop rows where CalltypeID is '11' or '12' (silence or noise)
        df = df[df.calltypeID != 11]
        df = df[df.calltypeID != 12]

        # Convert relevant rows to int
        df.twinID = df.twinID.astype(int)
        df.individualID = df.individualID.astype(int)

        # Convert individualID index to be independent of TwinID
        df.individualID = (df.twinID - 1) * 2 + df.individualID
        df.individualID -= 1  # Start index at 0, instead of 1

        # Reorder dataframe
        ordered_cols = [
            "audio_file",
            "start",
            "end",
            "duration",
            "calltypeID",
            "individualID",
        ]

        df = df[ordered_cols]

        # Reset index to account for dropped rows
        df.reset_index(drop=True, inplace=True)

        # Vocalization column
        df["vocID"] = df.index

        # Return
        return df

    def _load_audio_segment(self, index, audio_sample_path):
        start = self.filelist.start.iloc[index]
        end = self.filelist.end.iloc[index]
        signal, sr = librosa.load(
            audio_sample_path,
            sr=None,
            mono=True,
            # dtype="float32",
            offset=start,
            duration=end - start,
        )

        # Channels first like with torchaudio.load
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


if __name__ == "__main__":
    _ = IMVDataset()
