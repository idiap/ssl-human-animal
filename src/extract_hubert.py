# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import librosa
import lightning as L
import numpy as np
import rootutils
import torch

# from gridtk.tools import get_array_job_slice
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoModel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, manual_repeat_pad, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def get_model(upstream_model, cache_dir, device="cuda"):
    return AutoModel.from_pretrained(upstream_model, cache_dir=cache_dir).to(device)


def load_audio_segment(sample, target_sr):
    vocid = sample[0]
    file = sample[1]
    start = sample[2]
    end = sample[3]
    signal, _ = librosa.load(
        file,
        sr=target_sr,
        mono=True,
        offset=start,
        duration=end - start,
    )
    signal /= np.max(np.abs(signal))
    signal = torch.FloatTensor(signal)
    sig_len = len(signal)
    return vocid, signal, sig_len


def get_batch(samples, start, end, target_sr, min_len):
    wavs, vocids, wavs_len = [], [], []
    end_id = min(end, len(samples))
    for index in range(start, end_id):
        vocid, wav, sig_len = load_audio_segment(samples[index], target_sr)
        wavs.append(wav)
        vocids.append(vocid)
        wavs_len.append(sig_len)

    padded_wavs = collate(wavs, wavs_len, min_len)
    return padded_wavs, wavs_len, vocids


# def collate(wavs, padding_value: int = 0):
#     from torch.nn.utils.rnn import pad_sequence

#     padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=padding_value)
#     return padded_wavs


def collate(wavs, wavs_len, min_len):
    max_len = max(min_len, max(wavs_len))
    padded_wavs = torch.vstack([manual_repeat_pad(wav, max_len) for wav in wavs])
    return padded_wavs


def get_features(model, x, device, LEN_THRESHOLD, return_attentions):

    with torch.no_grad():

        # If batch size is 1 and sequence length is less than threshold
        if x.shape[-1] < LEN_THRESHOLD:
            out = model(
                x.to(device),
                output_hidden_states=True,
                output_attentions=return_attentions,
            )
            features = out.hidden_states
            features = torch.stack(features, dim=1)
            if return_attentions:
                attentions = out.attentions
                attentions = torch.stack(attentions, dim=1)
            else:
                attentions = None
            del out
            del x

        else:
            # Break up into smaller chunks and submit each to the model, then average
            num_chunks = x.shape[-1] // LEN_THRESHOLD
            # Iterate chunk by chunk
            features_lst = []
            attentions_lst = []
            for j in range(num_chunks):
                chunk = x[:, j * LEN_THRESHOLD : (j + 1) * LEN_THRESHOLD]
                outputs = model(
                    chunk.to(device),
                    output_hidden_states=True,
                    output_attentions=return_attentions,
                )
                del chunk
                features = outputs.hidden_states
                features = torch.stack(features, dim=1)
                features_lst.append(features.detach().cpu())
                del features
                # Stack
                if return_attentions:
                    attentions = outputs.attentions
                    attentions = torch.stack(attentions, dim=1)
                    attentions_lst.append(attentions)
                    del attentions
                del outputs

            # Handle the last batch if it exists
            if x.shape[-1] % LEN_THRESHOLD != 0:
                last_chunk = x[:, num_chunks * LEN_THRESHOLD :]
                outputs = model(
                    last_chunk.to(device),
                    output_hidden_states=True,
                    output_attentions=True,
                )
                del last_chunk
                features = outputs.hidden_states
                features = torch.stack(features, dim=1)
                features_lst.append(features.detach().cpu())
                del features
                if return_attentions:
                    attentions = outputs.attentions
                    attentions = torch.stack(attentions, dim=1)
                    attentions_lst.append(attentions)
                    del attentions
                del outputs
            del x

            # Concatenate along the time dimension
            features = torch.cat(
                features_lst, dim=2
            )  # Concatenate along the time dimension
            if return_attentions:
                attentions = torch.cat(
                    attentions_lst, dim=3
                )  # Concatenate along the time dimension
            else:
                attentions = None

            # # Average along the concatenated dimension
            # features = features.mean(dim=2)
            # if return_attentions:
            #     attentions = attentions.mean(dim=3)
            # else:
            #     attentions = None

        return features, attentions


def get_features_original(model, padded_wavs, device, return_attentions):

    with torch.no_grad():

        # Following returns a tuple of length = number of layers
        out = model(
            padded_wavs.to(device),
            output_hidden_states=True,
            output_attentions=return_attentions,
        )
        features = out.hidden_states
        if return_attentions:
            attentions = out.attentions
        del out
        # Reshape to (batch_size, num_layers, seq_len, hidden_size)
        features = torch.stack(features, dim=1)
        if return_attentions:
            attentions = torch.stack(attentions, dim=1)
        else:
            attentions = None
    return features, attentions


def unpad(hs, hs_len, index):
    hs_unpadded = []
    for h, lens in zip(hs, hs_len):
        l = lens[index]
        hs_unpadded.append(h[index, :l])
    return hs_unpadded


def pack_features(features, vocids, features_dict):
    num_inputs = len(vocids)
    for input_id in range(num_inputs):
        wav_id = str(vocids[input_id])
        # hs = unpad(features, features_len, input_id)
        features_dict[wav_id] = features[input_id]
    return features_dict


def pack_detach_features(features, vocids, features_dict):
    num_inputs = len(vocids)
    for input_id in range(num_inputs):
        wav_id = str(vocids[input_id])
        # hs = unpad(features, features_len, input_id)
        features_dict[wav_id] = features[input_id].detach().cpu().numpy()
    return features_dict


def compute_functionals(features, stack):
    """
    Compute and concatenate the mean and std of the features per layer.
    """
    for vocid, embed in features.items():
        layer_list = []
        for i in range(embed.shape[0]):
            mean = torch.mean(embed[i], dim=0)
            std = torch.std(embed[i], dim=0)
            layer_list.append(torch.cat([mean, std]).detach().cpu().numpy())
        stack[str(vocid)] = np.vstack(layer_list)
    return stack


def extract_feats(
    data_name,
    samples,
    target_sr,
    model,
    batch_size,
    device,
    min_len,
    max_len,
    save_dir,
    save_dir_attn,
    m_name,
    return_attentions,
):
    feature_dict = {}
    attention_dict = {}
    stack = {}
    num_wavs = len(samples)

    # Iterate by mini-batches
    c = 0
    for bid in trange(0, len(samples), batch_size):
        end_id = min(bid + batch_size, num_wavs)

        # Get batch
        pwavs, _, vocids = get_batch(samples, bid, end_id, target_sr, min_len)

        # Get features
        if data_name == "watkins":
            features, attentions = get_features(
                model, pwavs, device, max_len, return_attentions
            )
        else:
            features, attentions = get_features_original(
                model, pwavs, device, return_attentions
            )

        # Pack features
        features_dict = pack_features(features, vocids, feature_dict)

        # Compute functionals
        stack = compute_functionals(features_dict, stack)

        # Save
        log.info(f"Saving features {c} ...")
        save_path = os.path.join(save_dir, f"{m_name}_{c}.npz")
        np.savez(save_path, **stack)

        if attentions:
            attention_dict = pack_detach_features(attentions, vocids, attention_dict)
            del attentions
            save_path_attn = os.path.join(save_dir_attn, f"{m_name}_{c}.npz")
            np.savez(save_path_attn, **attention_dict)
            attention_dict = {}

        # Update and clear
        c += 1
        stack = {}
        feature_dict = {}

    # Return
    return stack


@task_wrapper
def extract(cfg: DictConfig):
    """Extracts features.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    save_dir = cfg.paths.feats_dir
    target_sr = cfg.sample_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model")
    # model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = get_model(os.path.join(cfg.owner, cfg.model), cfg.paths.hf_dir, device)

    model.eval()

    log.info("Checking save dir ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    save_dir_attn = os.path.join(save_dir + "_attentions")
    if not os.path.exists(save_dir_attn):
        os.makedirs(save_dir_attn, exist_ok=False)

    log.info("Extracting features ...")
    cols = ["vocID", "audio_file", "start", "end"]
    samples = datamodule.data.filelist[cols].values

    stack = extract_feats(
        cfg.data.data.name,
        samples,
        target_sr,
        model,
        cfg.batch_size,
        device,
        cfg.min_len,
        cfg.max_len,
        save_dir,
        save_dir_attn,
        cfg.model,
        cfg.return_attentions,
    )

    log.info("Exiting ...")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    _ = extract(cfg)


if __name__ == "__main__":
    main()
