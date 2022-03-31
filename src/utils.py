from enum import Enum, auto
from typing import Any, Union
import numpy as np

import torch


def select_device(index: Union[str, int]):
    if isinstance(index, int):
        return torch.device('cuda', index)
    if isinstance(index, str):
        if str.isdecimal(index):
            return torch.device('cuda', int(index))
        if index == 'cuda':
            return torch.device('cuda', 0)
        if index == 'cpu':
            return torch.device('cpu')
    raise TypeError('unsupported device ordinal')


class Format(Enum):
    TORCH = auto()
    NUMPY = auto()
    CV2 = auto()


class ImageConverter:
    @staticmethod
    def convert(image: Any, src_fm: Format, dst_fm: Format):
        if src_fm == Format.TORCH and dst_fm == Format.NUMPY:
            assert len(image.size()) == 3, 'invalid image dimension'
            return torch2numpy(image)
        if src_fm == Format.NUMPY and dst_fm == Format.TORCH:
            assert len(image.shape) == 3, 'invalid image dimension'
            return numpy2torch(image)
        if src_fm == Format.CV2 and dst_fm == Format.NUMPY:
            assert len(image.shape) == 3, 'invalid image dimension'
            return cv22numpy(image)
        if src_fm == Format.NUMPY and dst_fm == Format.CV2:
            assert len(image.shape) == 3, 'invalid image dimension'
            return numpy2cv2(image)
        if src_fm == Format.TORCH and dst_fm == Format.CV2:
            assert len(image.size()) == 3, 'invalid image dimension'
            return torch2cv2(image)
        if src_fm == Format.CV2 and dst_fm == Format.TORCH:
            assert len(image.shape) == 3, 'invalid image dimension'
            return cv22torch(image)

def torch2numpy(image: torch.Tensor) -> np.ndarray:
    if image.requires_grad:
        image = image.detach()
    if image.is_cuda:
        image = image.cpu()
    return image.permute(1, 2, 0).numpy()

def numpy2torch(image: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(image)
    return image.permute(2, 0, 1)

def cv22numpy(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return image / 255.

def numpy2cv2(image: np.ndarray) -> np.ndarray:
    image *= 255.
    return image.astype(np.uint8)

def torch2cv2(image: torch.Tensor) -> np.ndarray:
    return numpy2cv2(torch2numpy(image))

def cv22torch(image: np.ndarray) -> torch.Tensor:
    return numpy2torch(cv22numpy(image))
