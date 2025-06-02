import pooch
import pytest

from wpodnet import WPODNet, load_wpodnet_from_checkpoint


@pytest.fixture(scope="session")
def wpodnet() -> WPODNet:
    checkpoint = pooch.retrieve(
        "https://github.com/Pandede/WPODNet-Pytorch/releases/download/1.0.0/wpodnet.pth",
        known_hash="ac9fded54614d01b3082dd4e3917a65d4720b77a3f468fa934dbd85c814d3d77",
    )
    return load_wpodnet_from_checkpoint(checkpoint)
