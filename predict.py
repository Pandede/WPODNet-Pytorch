import errno
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import torch

from wpodnet.backend import Predictor
from wpodnet.model import WPODNet
from wpodnet.stream import ImageStreamer

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
        '--scale',
        type=float,
        default=1.0,
        help='adjust the scaling ratio. default to 1.0.'
    )
    parser.add_argument(
        '--save-annotated',
        type=str,
        help='save the annotated image at the given folder'
    )
    parser.add_argument(
        '--save-warped',
        type=str,
        help='save the warped image at the given folder'
    )
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise ArgumentTypeError(message='scale must be greater than 0.0')

    if args.save_annotated is not None:
        save_annotated = Path(args.save_annotated)
        if not save_annotated.is_dir():
            raise FileNotFoundError(errno.ENOTDIR, 'No such directory', args.save_annotated)
    else:
        save_annotated = None

    if args.save_warped is not None:
        save_warped = Path(args.save_warped)
        if not save_warped.is_dir():
            raise FileNotFoundError(errno.ENOTDIR, 'No such directory', args.save_warped)
    else:
        save_warped = None

    # Prepare for the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WPODNet()
    model.to(device)

    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint)

    predictor = Predictor(model)

    streamer = ImageStreamer(args.source)
    for i, image in enumerate(streamer):
        prediction = predictor.predict(image, scaling_ratio=args.scale)

        print(f'Prediction #{i}')
        print('  bounds', prediction.bounds.tolist())
        print('  confidence', prediction.confidence)

        if save_annotated:
            annotated_path = save_annotated / Path(image.filename).name
            annotated = prediction.annotate()
            annotated.save(annotated_path)
            print(f'Saved the annotated image at {annotated_path}')

        if save_warped:
            warped_path = save_warped / Path(image.filename).name
            warped = prediction.warp()
            warped.save(warped_path)
            print(f'Saved the warped image at {warped_path}')

        print()
