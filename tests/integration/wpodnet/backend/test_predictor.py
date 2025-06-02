from typing import List, Tuple

import pytest
from PIL import Image

from wpodnet import Predictor, WPODNet


@pytest.fixture
def predictor(wpodnet: WPODNet) -> Predictor:
    return Predictor(wpodnet)


class TestPredictor:
    @pytest.mark.parametrize(
        "image_path, bounds",
        [
            (
                "docs/sample/original/03009.jpg",
                [(1384, 682), (1517, 614), (1502, 685), (1369, 753)],
            ),
            (
                "docs/sample/original/03016.jpg",
                [(567, 375), (643, 368), (643, 393), (567, 400)],
            ),
            (
                "docs/sample/original/03025.jpg",
                [(94, 162), (162, 171), (160, 186), (92, 177)],
            ),
        ],
    )
    def test_predict(
        self, predictor: Predictor, image_path: str, bounds: List[Tuple[int, int]]
    ):
        image = Image.open(image_path)
        prediction = predictor.predict(image)
        assert prediction.bounds == bounds
        assert prediction.confidence >= 0.9
