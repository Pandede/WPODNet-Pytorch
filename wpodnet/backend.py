from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import _get_perspective_coeffs, to_tensor

from .model import WPODNet


@dataclass(frozen=True)
class Prediction:
    """
    The prediction result from WPODNet.

    Attributes:
        bounds (List[Tuple[int, int]]): The bounding coordinates of the detected license plate. Must be a list of 4 points (x, y).
        confidence (float): The confidence score of the detection. Must be between 0.0 and 1.0.
    """

    bounds: List[Tuple[int, int]]
    confidence: float

    def __post_init__(self):
        if len(self.bounds) != 4:
            raise ValueError(
                f"expected bounds to have 4 points, got {len(self.bounds)} points"
            )
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def annotate(
        self,
        canvas: Image.Image,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: int = 1,
    ) -> None:  # pragma: no cover
        """
        Annotates the image with the bounding polygon.

        Args:
            canvas (PIL.Image.Image): The image to be annotated.
            fill (Optional[str]): The fill color for the polygon. Defaults to None.
            outline (Optional[str]): The outline color for the polygon. Defaults to None.
            width (int): The width of the outline. Defaults to 1.

        Note:
            The arguments `fill`, `outline`, and `width` are passed to the `ImageDraw.Draw.polygon` method.
            See https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html#PIL.ImageDraw.ImageDraw.polygon.
        """
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(self.bounds, fill=fill, outline=outline, width=width)

    def warp(self, canvas: Image.Image) -> Image.Image:  # pragma: no cover
        """
        Warps the image with perspective based on the bounding polygon.

        Args:
            canvas (PIL.Image.Image): The image to be warped.

        Returns:
            PIL.Image.Image: The warped image.
        """
        coeffs = _get_perspective_coeffs(
            startpoints=self.bounds,
            endpoints=[
                (0, 0),
                (canvas.width, 0),
                (canvas.width, canvas.height),
                (0, canvas.height),
            ],
        )
        return canvas.transform(
            (canvas.width, canvas.height), Image.Transform.PERSPECTIVE, coeffs
        )


Q = np.array(
    [
        [-0.5, 0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
    ]
)


class Predictor:
    """A wrapper class for WPODNet to make predictions."""

    def __init__(self, wpodnet: WPODNet) -> None:
        """
        Args:
            wpodnet (WPODNet): The WPODNet model to use for prediction.
        """
        self.wpodnet = wpodnet
        self.wpodnet.eval()

    def _resize_to_fixed_ratio(
        self, image: Image.Image, dim_min: int, dim_max: int
    ) -> Image.Image:
        h, w = image.height, image.width

        wh_ratio = max(h, w) / min(h, w)
        side = int(wh_ratio * dim_min)
        bound_dim = min(side + side % self.wpodnet.stride, dim_max)

        factor = bound_dim / max(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)

        # Ensure the both width and height are the multiply of `self.wpodnet.stride`
        reg_w_mod = reg_w % self.wpodnet.stride
        if reg_w_mod > 0:
            reg_w += self.wpodnet.stride - reg_w_mod

        reg_h_mod = reg_h % self.wpodnet.stride
        if reg_h_mod > 0:
            reg_h += self.wpodnet.stride - reg_h_mod

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
        probs = np.squeeze(probs.cpu().numpy())[0]  # (grid_h, grid_w)
        affines = np.squeeze(affines.cpu().numpy())  # (6, grid_h, grid_w)

        return probs, affines

    def _get_max_anchor(self, probs: np.ndarray) -> Tuple[int, int]:
        return np.unravel_index(probs.argmax(), probs.shape)

    def _get_bounds(
        self,
        affines: np.ndarray,
        anchor_y: int,
        anchor_x: int,
        scaling_ratio: float = 1.0,
    ) -> np.ndarray:
        # Compute theta
        theta = affines[:, anchor_y, anchor_x]
        theta = theta.reshape((2, 3))
        theta[0, 0] = max(theta[0, 0], 0.0)
        theta[1, 1] = max(theta[1, 1], 0.0)

        # Convert theta into the bounding polygon
        bounds = np.matmul(theta, Q) * self.wpodnet.scale_factor * scaling_ratio

        # Normalize the bounds
        _, grid_h, grid_w = affines.shape
        bounds[0] = (bounds[0] + anchor_x + 0.5) / grid_w
        bounds[1] = (bounds[1] + anchor_y + 0.5) / grid_h

        return np.transpose(bounds)

    def predict(
        self,
        image: Image.Image,
        scaling_ratio: float = 1.0,
        dim_min: int = 512,
        dim_max: int = 768,
    ) -> Prediction:
        """
        Detect license plate in the image.

        Args:
            image (Image.Image): The image to be detected.
            scaling_ratio (float): The scaling ratio of the resulting bounding polygon. Default to 1.0.
            dim_min (int): The minimum dimension of the resized image. Default to 512
            dim_max (int): The maximum dimension of the resized image. Default to 768

        Returns:
            Prediction: The prediction result with highest confidence.
        """
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
            bounds=[(x, y) for x, y in np.int32(bounds).tolist()],
            confidence=max_prob.item(),
        )
