import torch


class BasicConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn_layer = torch.nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        self.act_layer = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return self.act_layer(x)


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = BasicConvBlock(channels, channels)
        self.sec_layer = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_layer = torch.nn.BatchNorm2d(channels, momentum=0.99, eps=0.001)
        self.act_layer = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.sec_layer(h)
        h = self.bn_layer(h)
        return self.act_layer(x + h)


class WPODNet(torch.nn.Module):
    """
    WPODNet in PyTorch.

    The original architecture is built in Keras: https://github.com/sergiomsilva/alpr-unconstrained/blob/master/create-model.py
    """

    # https://github.com/sergiomsilva/alpr-unconstrained/blob/master/src/keras_utils.py#L43-L44
    stride = 16  # net_stride
    scale_factor = 7.75  # side

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            BasicConvBlock(3, 16),
            BasicConvBlock(16, 16),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(16, 32),
            ResBlock(32),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(32, 64),
            ResBlock(64),
            ResBlock(64),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(64, 64),
            ResBlock(64),
            ResBlock(64),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(64, 128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )
        self.prob_layer = torch.nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.bbox_layer = torch.nn.Conv2d(128, 6, kernel_size=3, padding=1)

        # Registry a dummy tensor for retrieve the attached device
        self.register_buffer("dummy", torch.Tensor(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def forward(self, image: torch.Tensor):
        feature: torch.Tensor = self.backbone(image)
        probs: torch.Tensor = self.prob_layer(feature)
        probs = torch.softmax(probs, dim=1)
        affines: torch.Tensor = self.bbox_layer(feature)

        return probs, affines
