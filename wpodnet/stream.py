from pathlib import Path
from typing import Generator, Union

from PIL import Image


class ImageStreamer:
    def __init__(self, image_or_folder: Union[str, Path]):
        path = Path(image_or_folder)
        self.generator = self._get_image_generator(path)

    def _get_image_generator(self, path: Path) -> Generator[Image.Image, None, None]:
        if path.is_file():
            image_paths = [path] if self._is_image_file(path) else []
        elif path.is_dir():
            image_paths = [
                p
                for p in path.rglob('**/*')
                if self._is_image_file(p)
            ]
        else:
            raise TypeError(f'Invalid path to images {path}')

        for p in image_paths:
            yield Image.open(p)

    def _is_image_file(self, path: Path) -> bool:
        try:
            image = Image.open(path)
            image.verify()
            return True
        except Exception:
            return False

    def __iter__(self):
        return self.generator
