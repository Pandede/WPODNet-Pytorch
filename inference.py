import argparse
from pathlib import Path

import cv2

from src.backend import DetectBackend
from src.utils import Format, ImageConverter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='inference image file')
    parser.add_argument('--weight', default='./weight/wpodnet.pth', type=str, help='torch weight file')
    parser.add_argument('--device', default='cpu', type=str, help='inference on "cuda" or "cpu"')
    parser.add_argument('--output', default='./output', type=str, help='output folder for saving')
    opt = parser.parse_args()

    backend = DetectBackend(Path(opt.weight), opt.device)

    image = cv2.imread(opt.source)
    image = ImageConverter.convert(image, Format.CV2, Format.TORCH)
    result, prob = backend.inference(image)
    result = ImageConverter.convert(result, Format.TORCH, Format.CV2)
    output_fp = Path(opt.output) / Path(opt.source).name
    cv2.imwrite(str(output_fp), result)
    print(prob)
