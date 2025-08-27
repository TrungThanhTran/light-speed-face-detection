import torch.nn as nn


class ConvDPUnit(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        withBNRelu=True,
        use_se: bool = False,            # <-- NEW
        se_reduction: int = 8            # <-- NEW
    ):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)

        # --- NEW: optional SE after DW conv, before BN/ReLU ---
        self.se = SELayer(out_channels, se_reduction) if use_se else None

        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)        # pointwise 1x1
        x = self.conv2(x)        # depthwise 3x3
        if self.se is not None:  # <-- NEW: SE recalibration (no shape change)
            x = self.se(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class Conv_head(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 use_se: bool = False, se_reduction: int = 8):   # <-- NEW
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        # pass flags into ConvDPUnit
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True,
                                use_se=use_se, se_reduction=se_reduction)  # <-- CHANGED
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withBNRelu=True,
                 use_se: bool = False, se_reduction: int = 8):            # <-- NEW
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True,
                                use_se=False, se_reduction=se_reduction)    # keep first clean
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu,
                                use_se=use_se, se_reduction=se_reduction)   # <-- SE here

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SELayer(nn.Module):
    """Squeeze-and-Excitation for channel re-weighting (cheap)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w