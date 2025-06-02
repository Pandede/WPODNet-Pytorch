import pytest

from wpodnet import Prediction


class TestPrediction:
    def test_validation(self):
        # Valid
        Prediction(bounds=[(1, 2), (3, 4), (5, 6), (7, 8)], confidence=0.95)

        # Invalid bounds length
        with pytest.raises(ValueError):
            Prediction(bounds=[(1, 2), (3, 4), (5, 6)], confidence=0.95)

        # Invalid confidence value
        with pytest.raises(ValueError):
            Prediction(bounds=[(1, 2), (3, 4), (5, 6), (7, 8)], confidence=1.01)
