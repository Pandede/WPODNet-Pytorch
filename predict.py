import errno
import os.path
from argparse import ArgumentParser

from PIL import Image
import torch
from wpodnet.backend import Predictor
from wpodnet.model import WPODNet


def has_parent_folder(p: str) -> bool:
    return os.path.isdir(os.path.dirname(p))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'source',
        type=str,
        help='the path to the image'
    )
    parser.add_argument(
        '-w', '--weight',
        type=str,
        required=True,
        help='the path to the model weight'
    )
    parser.add_argument(
        '--save-annotated',
        type=str,
        help='save the annotated image to the given filepath'
    )
    parser.add_argument(
        '--save-warped',
        type=str,
        help='save the warped image to the given filepath'
    )
    args = parser.parse_args()

    if not has_parent_folder(args.save_annotated):
        raise FileNotFoundError(errno.ENOENT, 'No such directory', f"'{args.save_annotated}'")

    if not has_parent_folder(args.save_warped):
        raise FileNotFoundError(errno.ENOENT, 'No such directory', f"'{args.save_warped}'")

    image = Image.open(args.source)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WPODNet()
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint)

    predictor = Predictor(model)
    prediction = predictor.predict(image)

    print('Prediction')
    print('  bounds', prediction.bounds.tolist())
    print('  confidence', prediction.confidence)

    if args.save_annotated is not None:
        prediction.annotate(args.save_annotated)
        print(f'Saved the annotated image at {args.save_annotated}')

    if args.save_warped is not None:
        prediction.warp(args.save_warped)
        print(f'Saved the warped image at {args.save_warped}')
