from pathlib import Path
from typing import List, Tuple

import torch
from model.wpodnet import WPODNet
from torchvision.transforms.functional import perspective, resize

from .utils import select_device


@torch.no_grad()
class DetectBackend:
    def __init__(self, weight: Path, device: str, output_size: Tuple[int] = (60, 208)):
        self.device = select_device(device)
        self.backbone = self.__load_backbone(weight)
        self.output_h, self.output_w = output_size
        self.affine_base = torch.Tensor([
            [-.5, .5, .5, -.5],
            [-.5, -.5, .5, .5],
            [1., 1., 1., 1.]
        ], device=self.device)

    def __load_backbone(self, weight: Path):
        model = WPODNet().to(self.device)
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval()
        return model

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        h = image.size(1)
        w = image.size(2)

        ratio = max(h, w) / min(h, w)
        side = int(ratio * 288.)
        bound_dim = min(side + side % (2**4), 608.)

        factor = bound_dim / min(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)
        resized_image = resize(image, (reg_h, reg_w))
        return resized_image

    def postprocess(self, probs: torch.Tensor, affines: torch.Tensor, pad_ratio: float = 1.0):
        ry, rx = probs.size(0), probs.size(1)
        hy, hx = torch.nonzero(probs == torch.max(probs))[0]
        theta = affines[:, hy, hx].view(2, 3)
        theta[0, 0] = max(theta[0, 0], 0.)
        theta[1, 1] = max(theta[1, 1], 0.)

        mn = torch.Tensor([[hx + .5], [hy + .5]])
        grid_pts = torch.matmul(theta, self.affine_base) * 7.75 * pad_ratio + mn
        grid_pts[0] /= rx
        grid_pts[1] /= ry

        prob = probs[hy, hx]
        return grid_pts.t(), prob

    def warp_perspective(self, image: torch.Tensor, start_pts: List[List[int]], end_pts: List[List[int]]) -> torch.Tensor:
        return perspective(image, start_pts, end_pts)

    def get_corners(self, image_size: Tuple[int]):
        h, w = image_size
        return torch.Tensor([[0, 0], [w, 0], [w, h], [0, h]], device=self.device)

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        img_tensor = self.preprocess(image).unsqueeze_(0)
        probs, affines = self.backbone(img_tensor)
        probs = probs.squeeze_()[0]
        affines = affines.squeeze_()
        grid_pts, prob = self.postprocess(probs, affines)

        img_h, img_w = image.size(1), image.size(2)
        grid_pts[:, 0] *= img_w
        grid_pts[:, 1] *= img_h
        grid_pts = grid_pts.contiguous()
        end_pts = self.get_corners((self.output_h, self.output_w))
        warped_image = self.warp_perspective(image, grid_pts, end_pts)
        return warped_image[:, :self.output_h, :self.output_w], prob
