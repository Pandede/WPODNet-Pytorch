from pathlib import Path
from typing import Union

import torch

from .backend import Prediction, Predictor
from .model import WPODNet


def load_wpodnet_from_checkpoint(ckpt_path: Union[str, Path]) -> WPODNet:
    """
    Load a pre-trained WPOD-NET model from a checkpoint file.

    Args:
        ckpt_path (Union[str, Path]): The path to the checkpoint file.

    Returns:
        WPODNet: The WPOD-NET model with pretrained weights loaded from the checkpoint.
    """
    model = WPODNet()

    # Load the state dictionary from the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint)

    return model


__all__ = ["Prediction", "Predictor", "WPODNet", "load_wpodnet_from_checkpoint"]
