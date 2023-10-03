from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import (_get_perspective_coeffs,
                                               to_tensor)

from .model import WPODNet


class Prediction:
    def __init__(self, image: Image.Image, bounds: np.ndarray, confidence: float):
        self.image = image
        self.bounds = bounds
        self.confidence = confidence

    def _get_perspective_coeffs(self, width: int, height: int) -> List[float]:
        # Get the perspective matrix
        src_points = self.bounds.tolist()
        dst_points = [[0, 0], [width, 0], [width, height], [0, height]]
        return _get_perspective_coeffs(src_points, dst_points)

    def annotate(self, outline: str = 'red', width: int = 3) -> Image.Image:
        canvas = self.image.copy()
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(
            [(x, y) for x, y in self.bounds],
            outline=outline,
            width=width
        )
        return canvas

    def warp(self, width: int = 208, height: int = 60) -> Image.Image:
        # Get the perspective matrix
        coeffs = self._get_perspective_coeffs(width, height)
        warped = self.image.transform((width, height), Image.PERSPECTIVE, coeffs)
        return warped


class Predictor:
    _q = np.array([
        [-.5, .5, .5, -.5],
        [-.5, -.5, .5, .5],
        [1., 1., 1., 1.]
    ])
    _scaling_const = 7.75
    _stride = 16

    def __init__(self, wpodnet: WPODNet):
        self.wpodnet = wpodnet
        self.wpodnet.eval()

    def _resize_to_fixed_ratio(self, image: Image.Image, dim_min: int, dim_max: int) -> Image.Image:
        h, w = image.height, image.width

        wh_ratio = max(h, w) / min(h, w)
        side = int(wh_ratio * dim_min)
        bound_dim = min(side + side % self._stride, dim_max)

        factor = bound_dim / min(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)

        # Ensure the both width and height are the multiply of `self._stride`
        reg_w += self._stride - reg_w % self._stride
        reg_h += self._stride - reg_h % self._stride

        return image.resize((reg_w, reg_h))

    def _to_torch_image(self, image: Image.Image) -> torch.Tensor:
        tensor = to_tensor(image)
        return tensor.unsqueeze_(0)

    def _inference(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            probs, affines = self.wpodnet.forward(image)

        # Convert to squeezed numpy array
        # grid_w: The number of anchors in row
        # grid_h: The number of anchors in column
        probs = np.squeeze(probs.cpu().numpy())[0]     # (grid_h, grid_w)
        affines = np.squeeze(affines.cpu().numpy())  # (6, grid_h, grid_w)

        return probs, affines

    def _get_max_anchor(self, probs: np.ndarray) -> Tuple[int, int]:
        return np.unravel_index(probs.argmax(), probs.shape)

    def _get_bounds(self, affines: np.ndarray, anchor_y: int, anchor_x: int, scaling_ratio: float = 1.0) -> np.ndarray:
        # Compute theta
        theta = affines[:, anchor_y, anchor_x]
        theta = theta.reshape((2, 3))
        theta[0, 0] = max(theta[0, 0], 0.0)
        theta[1, 1] = max(theta[1, 1], 0.0)

        # Convert theta into the bounding polygon
        bounds = np.matmul(theta, self._q) * self._scaling_const * scaling_ratio

        # Normalize the bounds
        _, grid_h, grid_w = affines.shape
        bounds[0] = (bounds[0] + anchor_x + .5) / grid_w
        bounds[1] = (bounds[1] + anchor_y + .5) / grid_h

        return np.transpose(bounds)

    def predict(self, image: Image.Image, scaling_ratio: float = 1.0, dim_min: int = 288, dim_max: int = 608) -> Prediction:
        orig_h, orig_w = image.height, image.width

        # Resize the image to fixed ratio
        # This operation is convienence for setup the anchors
        resized = self._resize_to_fixed_ratio(image, dim_min=dim_min, dim_max=dim_max)
        resized = self._to_torch_image(resized)
        resized = resized.to(self.wpodnet.device)

        # Inference with WPODNet
        # probs: The probability distribution of the location of license plate
        # affines: The predicted affine matrix
        probs, affines = self._inference(resized)

        # Get the theta with maximum probability
        max_prob = np.amax(probs)
        anchor_y, anchor_x = self._get_max_anchor(probs)
        bounds = self._get_bounds(affines, anchor_y, anchor_x, scaling_ratio)

        bounds[:, 0] *= orig_w
        bounds[:, 1] *= orig_h

        return Prediction(
            image=image,
            bounds=bounds.astype(np.int32),
            confidence=max_prob.item()
        )
