# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def compile(cfg: DictConfig):
    """
    Compiles extracts features into a single dict.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    feats_dir = cfg.paths.feats_dir
    attn_dir = cfg.paths.feats_dir + "_attentions"
    log.info(f"Compiling features ...")
    save_name = os.path.join(feats_dir, f"all_{cfg.model}.npz")
    save_name_attn = os.path.join(attn_dir, f"all_{cfg.model}.npz")

    log.info(feats_dir)
    if os.path.exists(feats_dir):
        files = sorted(glob.glob(os.path.join(feats_dir, f"{cfg.model}*.npz")))
        if files:
            arrays = []
            for file in tqdm(files):
                arrays.append(dict(np.load(file)))
                # Convert list of dicts to a single dict
                stack = {k: v for e in arrays for k, v in e.items()}

            log.info(f"Saving ...")
            np.savez(save_name, **stack)

            log.info("Deleting old files ...")
            for file in files:
                os.remove(file)

    # log.info(attn_dir)
    # if os.path.exists(attn_dir):
    #     files = sorted(glob.glob(os.path.join(attn_dir, f"{cfg.model}*.npz")))
    #     if files:
    #         arrays = []
    #         for file in tqdm(files):
    #             arrays.append(dict(np.load(file)))
    #             # Convert list of dicts to a single dict
    #             stack = {k: v for e in arrays for k, v in e.items()}

    #         log.info(f"Saving ...")
    #         np.savez(save_name_attn, **stack)

    #         log.info("Deleting old files ...")
    #         for file in files:
    #             os.remove(file)

    log.info("Exiting ...")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    _ = compile(cfg)


if __name__ == "__main__":
    main()
