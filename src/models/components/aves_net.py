# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
import urllib.request

import fairseq
import torch
from torch import nn

class AvesClassifier(nn.Module):
    """The AVES classifier."""

    def __init__(self, model_path):

        super().__init__()
        # Check if model exists at path, else download
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Downloading from ...")
            dl_url = (
                "https://storage.googleapis.com/esp-public-files/aves/aves-base-bio.pt"
            )
            urllib.request.urlretrieve(dl_url, model_path)
            print("Download complete.")

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        self.layer_outputs = []

        def get_activation(layer_name):
            def hook(model, input, output):
                if layer_name.startswith("encoder_layer"):
                    out = output[0]
                elif layer_name.startswith("embedding_layer"):
                    out = output.permute(1, 0, 2)
                self.layer_outputs.append(out)
            return hook
        
        # Register a hook for the embedding layer
        self.model.post_extract_proj.register_forward_hook(
            get_activation("embedding_layer")
        )

        # Register hooks to the 12 transformer encoder layers
        for i, layer in enumerate(self.model.encoder.layers):
            layer.register_forward_hook(get_activation(f"encoder_layer_{i}"))
    
    def forward(self, x, output_attentions=False):
        # Reset layer outputs
        self.layer_outputs = []

        # Forward pass
        _ = self.model.extract_features(x)


        # Ensure all tensors in self.layer_outputs have the same dimensions
        for i, tensor in enumerate(self.layer_outputs):
            if len(tensor.shape) == 2:  # Case for embedding layer (no explicit batch dim)
                self.layer_outputs[i] = tensor.unsqueeze(1)  # Add a batch dimension

        # Find the minimum sequence length across all layers
        min_seq_len = min(tensor.size(0) for tensor in self.layer_outputs)

        # Truncate all tensors to the minimum sequence length
        self.layer_outputs = [tensor[:min_seq_len, :, :] for tensor in self.layer_outputs]

        # Concatenate all layer outputs
        out = torch.stack(self.layer_outputs, dim=0).permute(2, 0, 1, 3)

        return out

        # Compute functionals
        # return torch.concat([out.mean(dim=2), out.max(dim=2).values], dim=-1)

