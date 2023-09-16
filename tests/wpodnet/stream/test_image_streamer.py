from pathlib import Path

import pytest
from PIL import Image

from wpodnet.stream import ImageStreamer


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    exts = {'.jpeg', '.png', '.bmp'}
    for ext in exts:
        image = Image.new('RGB', (10, 10))
        image_path = (tmp_path / 'image').with_suffix(ext)
        image.save(image_path)
    return tmp_path


@pytest.mark.usefixtures('image_folder')
class TestImageStreamer:
    def test_load_image_file(self, image_folder: Path):
        for image_path in image_folder.glob('**/*'):
            streamer = ImageStreamer(image_path)
            images = list(streamer)
            assert len(images) == 1
            assert f'.{images[0].format.lower()}' == image_path.suffix

    def test_load_image_folder(self, image_folder: Path):
        streamer = ImageStreamer(image_folder)
        images = list(streamer)
        assert len(images) == 3

        # Add a non-image file
        (image_folder / 'text.doc').touch()
        streamer = ImageStreamer(image_folder)
        images = list(streamer)
        assert len(images) == 3

    def test_load_invalid_image(self, tmp_path: Path):
        doc_file = tmp_path / 'image.doc'

        with pytest.raises(TypeError, match='Invalid path to images'):
            list(ImageStreamer(doc_file))
